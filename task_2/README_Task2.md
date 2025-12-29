# Report Exploration: Trace Replay untuk PlanQualityMetric saat Agent Dipanggil via API (Stream Logs Only)

Author: Anthony Edbert Feriyanto

Tanggal: 29 Desember 2025

## 1. Latar Belakang Masalah

Saya menjalankan agent melalui API. Output yang saya dapat hanya stream logs, seperti token streaming, tool call events, dan final answer. Saya tidak bisa mengandalkan tracing otomatis berbasis decorator @observe untuk memonitor eksekusi internal agent, terutama karena LLM yang dipakai adalah Gemini via API key sehingga saya tidak punya integrasi tracing end to end seperti observability tool pada umumnya.

Saya tetap ingin memakai metric PlanQualityMetric dari DeepEval. Masalahnya, PlanQualityMetric membutuhkan trace sebagai input. Saat saya mencoba membuat Trace manual dan meng-inject ke trace_manager, metric tetap mengembalikan pesan “no plans to evaluate”.

## 2. Tujuan Exploration

1. Memahami bentuk trace yang dibentuk oleh decorator @observe.
2. Memastikan apakah saya bisa mereplikasi trace tersebut tanpa @observe, cukup dengan stream logs dari API.
3. Menemukan strategi agar metric PlanQualityMetric tetap bisa dipakai walau agent dipanggil via API dan hanya ada stream logs.
4. Menyusun alternatif jika PlanQualityMetric tidak dapat dipakai tanpa tracing pipeline bawaan.

## 3. Temuan Utama

### 3.1 Manual injection Trace tanpa @observe tidak bekerja untuk PlanQualityMetric

Saya mencoba membangun Trace object sendiri, lengkap dengan LlmSpan untuk planning dan ToolSpan untuk tool calls, lalu memasukkannya ke trace_manager.traces.

Hasilnya, PlanQualityMetric tetap gagal mendeteksi plan, dengan alasan:
There are no plans to evaluate within the trace of your agent's execution.

Interpretasi saya:
PlanQualityMetric tidak hanya membaca trace_manager.traces. Metric ini terikat pada mekanisme tracing pipeline internal DeepEval. Artinya, trace harus terbentuk melalui jalur yang “resmi” menurut DeepEval (biasanya lewat @observe atau evals_iterator), bukan sekadar manual assignment ke trace_manager.

### 3.2 DeepEval butuh tracing pipeline aktif, bukan hanya data trace

Dari eksperimen, saya simpulkan PlanQualityMetric adalah trace-dependent metric. Metric ini tidak dirancang untuk dipakai standalone hanya dengan objek Trace. Justru mengandalkan proses internal ketika trace dibuat pada waktu evaluasi.

Implikasinya:
Kalau saya tidak mengaktifkan tracing pipeline, PlanQualityMetric bawaan tidak bisa dipakai. Ini menjelaskan kenapa manual trace injection terus gagal walau struktur Trace sudah terlihat benar.

### 3.3 “Replay run” adalah cara paling masuk akal untuk kasus stream logs

Saya menggunakan pendekatan replay:

1. Saya jalankan agent seperti biasa dan mengumpulkan stream logs.
2. Saya memanggil fungsi wrapper yang di-decorate @observe, tapi fungsi ini tidak memanggil Gemini lagi.
3. Di dalam wrapper, saya “memutar ulang” logs itu menjadi span, menggunakan observed spans untuk planning dan tool calls.

Dengan pendekatan ini:
DeepEval benar-benar membangun trace yang dianggap valid. PlanQualityMetric akhirnya berjalan dan memberikan score serta reason.

Saya berhasil mendapatkan output score dan reason yang masuk akal, misalnya:
PlanQualityMetric score: 0.25
Reason: Plan is missing quantity, destination, payment method, email, etc.

Ini membuktikan tracing pipeline-nya hidup, trace terbentuk, dan metric berjalan.

## 4. Pembahasan Pendekatan Solusi

Saya menguji dua pendekatan utama: Solusi A dan Solusi B.

### 4.1 Solusi A: @observe hanya sebagai container tracer, bukan tracer LLM

Ide utamanya:
Saya tetap memakai @observe, tapi hanya untuk mengaktifkan tracing pipeline DeepEval. Saya tidak men-trace Gemini API call. Saya hanya “replay” log stream yang sudah ada menjadi spans observed.

Alur:

1. Mengumpulkan logs (planning, tool calls, final output) dari stream.
2. Parse log menjadi event standar: planning, tool, final.
3. Panggil agent_replay_from_events yang ter-decorate @observe.
4. Di dalamnya:

   * membuat span planning bernama thinking bertipe agent
   * membuat span tool untuk setiap tool call
   * update_current_trace dengan input, output, tools_called
5. Menjalankan PlanQualityMetric lewat evals_iterator.

Catatan penting:
Plan yang saya masukkan sering kali bukan plan asli dari model. Karena stream logs sering tidak memuat “thinking/planning” internal. Jadi saya membuat plan sintetis berdasarkan user_input dan urutan tool calls.

Hasil:
PlanQualityMetric berjalan dan memberikan score.

Tradeoff:
Skor plan menilai plan sintetis atau plan rekonstruksi, bukan “plan internal” yang benar-benar dipikir LLM.

### 4.2 Solusi B: Tanpa observe sama sekali, buat custom metric dari logs

Ide utamanya:
Saya tidak memakai PlanQualityMetric bawaan. Saya membuat metric baru yang meniru cara kerja PlanQuality:

1. ambil task dari input
2. ambil plan dari logs, atau rekonstruksi plan dari tool calls
3. minta LLM evaluator menilai kualitas plan terhadap task
4. hasilkan score 0..1 dan reason

Alur:

1. Input: stream logs saja.
2. Extract plan:

   * kalau ada planning event, pakai itu
   * kalau tidak ada, buat semantic plan dari tool names + detail task
3. Buat tools summary JSON untuk konteks judge.
4. Prompt Gemini untuk mengeluarkan JSON {score, reason}.
5. Parse JSON dengan safe_json_extract.
6. Return score, reason, dan pass/fail berdasar threshold.

Kelebihan:
Tidak butuh tracing pipeline DeepEval dan cocok untuk environment API-only.

Kekurangan:
Ini bukan PlanQualityMetric built-in. Ini custom metric.
Konsistensi skor sangat bergantung pada prompt judge dan stabilitas model evaluator.

## 5. Observasi dari Eksperimen Skor

Saat saya menjalankan Solusi A pertama kali, skor plan rendah (0.25). Alasannya bukan karena trace gagal, tapi karena plan yang saya log terlalu generik.

Plan generik:
Check inventory → Apply discount → Calculate shipping → Process payment → Send confirmation

Judge menilai plan ini tidak menyebut detail task seperti:
quantity 2
destination Jakarta
payment method credit card
email tujuan

Perbaikan yang efektif:
Saya mengganti planning yang generik menjadi plan sintetis yang memasukkan detail task.
Misalnya:

1. Parse request dan ekstrak product, qty, destination, discount, payment, email
2. check_inventory untuk qty 2
3. apply_discount WELCOME10
4. calculate_shipping ke Jakarta
5. process_payment pakai credit card
6. send_confirmation_email ke email user

Setelah plan dibuat spesifik, saya expect skor PlanQuality akan naik karena plan sudah selaras dengan task.

## 6. Kesimpulan

1. PlanQualityMetric DeepEval membutuhkan tracing pipeline aktif. Manual trace injection ke trace_manager saja tidak cukup.
2. Cara paling praktis untuk tetap memakai PlanQualityMetric saat agent dipanggil via API adalah replay logs ke dalam wrapper @observe.
3. Jika saya benar-benar tidak ingin memakai observe, maka saya harus membuat custom metric yang menilai plan dari stream logs secara langsung.
4. Tanpa planning text eksplisit dari agent, saya tidak bisa mengakses “plan internal” model. Yang bisa saya nilai hanya plan eksplisit (jika ada di stream) atau plan rekonstruksi dari tool calls.

## 7. Rekomendasi

### Rekomendasi A (paling sesuai untuk kebutuhan metric built-in)

Gunakan Solusi A:

* Pakai @observe hanya sebagai container tracing.
* Replay stream logs menjadi spans.
* Jalankan PlanQualityMetric lewat evals_iterator.
* Buat plan sintetis yang memasukkan detail task agar scoring tidak jatuh.

Ini paling mendekati harapan manajer jika targetnya “tetap pakai PlanQualityMetric bawaan” walau agent dipanggil via API.

### Rekomendasi B (kalau ingin benar-benar bebas dari observe)

Gunakan Solusi B:

* Buat metric custom PlanQualityFromLogsMetric.
* Extract plan dari logs atau rekonstruksi dari tool calls.
* Judge menggunakan Gemini evaluator.

Solusi ini cocok jika constraint sistem benar-benar tidak mengizinkan observe, atau jika tidak ingin mengikat diri pada tracing pipeline DeepEval.

### Rekomendasi tambahan untuk kualitas evaluasi

Kalau memungkinkan, saya akan mengubah prompt agent supaya selalu mengeluarkan plan eksplisit di awal, minimal 3 sampai 6 langkah. Dengan begitu:

* plan yang dievaluasi lebih mendekati “plan asli”
* skor PlanQuality lebih meaningful
* replay trace jadi lebih akurat

## 8. Lampiran: Ringkasan Implementasi

Solusi A:

* StreamingLogCollector untuk kumpulkan planning, tool_call, final
* parse_events_from_openai_like_log untuk konversi log ke StreamEvent
* agent_replay_from_events (observed) untuk replay span dan update_current_trace
* PlanQualityMetric dijalankan via dataset.evals_iterator

Solusi B:

* StreamEvent sebagai format event standar
* extract_plan_text dengan fallback semantic plan dari tool calls
* PlanQualityFromLogsMetric (BaseMetric) memanggil Gemini judge, output JSON score dan reason
