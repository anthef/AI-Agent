# E-Commerce Order Processing Agent

Agent sederhana untuk eksperimen AI Agent evaluation dengan DeepEval. Agent ini memproses pesanan e-commerce dengan multiple dependent tools.

## Features

Agent ini memiliki 5 tools yang saling berhubungan:

1. **check_inventory** - Cek ketersediaan produk dan harga
2. **apply_discount** - Terapkan kode diskon
3. **calculate_shipping** - Hitung biaya pengiriman berdasarkan destinasi
4. **process_payment** - Proses pembayaran
5. **send_confirmation_email** - Kirim email konfirmasi

### Produk yang Tersedia

- `LAPTOP-001` - Business Laptop ($1,200)
- `MOUSE-002` - Wireless Mouse ($25)
- `KEYBOARD-003` - Mechanical Keyboard ($80)
- `MONITOR-004` - 4K Monitor ($450)
- `HEADSET-005` - Noise Cancelling Headset ($150)

### Kode Diskon

- `WELCOME10` - 10% off untuk pelanggan baru
- `SAVE50` - $50 off untuk pesanan di atas $500
- `VIP20` - 20% off untuk member VIP

### Destinasi Pengiriman

- Jakarta (1 hari)
- Bandung (2 hari)
- Surabaya (3 hari)
- Bali (4 hari)
- Singapore (5 hari)

## Installation

### 1. Clone atau Download Repository

```bash
cd AgentAI
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
```

### 3. Aktivasi Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Setup Environment Variables

File `.env` sudah ada dengan GEMINI_API_KEY. Pastikan file ini tidak di-commit ke Git (sudah ada di `.gitignore`).

## Usage

### Menjalankan Agent Secara Manual

```bash
python agent.py
```

### Menjalankan 3 Skenario Contoh

```bash
python examples.py
```

## Example Scenarios

### Scenario 1: Standard Order with Discount

**Input:**
```
I want to order 2 laptops to Jakarta, use discount code WELCOME10, 
pay with credit card, email: john.doe@example.com
```

**Expected Output:**
- Product: Business Laptop x 2
- Subtotal: $2,400
- Discount: -$240 (10%)
- Shipping: ~$15
- Total: ~$2,175
- Delivery: 1 day

### Scenario 2: VIP Order with International Shipping

**Input:**
```
Order 1 monitor and ship to Singapore, I have VIP20 discount code, 
payment via ewallet, send confirmation to vip.customer@company.com
```

**Expected Output:**
- Product: 4K Monitor x 1
- Subtotal: $450
- Discount: -$90 (20%)
- Shipping: ~$70
- Total: ~$430
- Delivery: 5 days

### Scenario 3: Bulk Order with Large Discount

**Input:**
```
I need 5 headsets delivered to Surabaya, apply SAVE50 discount, 
pay by bank transfer, email: procurement@company.id
```

**Expected Output:**
- Product: Noise Cancelling Headset x 5
- Subtotal: $750
- Discount: -$50 (fixed)
- Shipping: ~$70
- Total: ~$770
- Delivery: 3 days

## Project Structure

```
AgentAI/
├── agent.py              # Main agent implementation
├── tools.py              # Hardcoded e-commerce tools
├── llm.py                # Gemini LLM wrapper
├── examples.py           # 3 example scenarios
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not in Git)
├── .gitignore           # Git ignore file
└── README.md            # This file
```

