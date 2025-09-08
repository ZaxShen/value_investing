# TODO List

## 🤖 For AI Assistants

**IMPORTANT**: Rules for AI Tools and Assistants

Any AI assistant working with this repository MUST strictly adhere to the following guidelines:

### ✅ You CAN

- Read and analyze TODO items
- Discuss technical approaches
- Answer questions about features

---

### ❌ You CANNOT

- **NEVER IMPLEMENT** any TODO items
- Modify this file or any mentioned files
- Treat TODO items as tasks

---

### 📋 Key Points

- Human-only planning document
- AI = advisory role only
- **NO EXCEPTIONS**

---

## v1.8.0 TODO

### Apply more technical metrics for stock filtering

Target scripts:

- stock_filter.py

New Features:

- Add technical features to config files to filter stocks by tech metrics, for example, stocks with MACD near 0

---

## v1.9.0 TODO

### Use DB to cache data and extend maximum available data

- Use DB like PostgreSQL to store batch data from akshare.
- As 东方财富 has maximum data period like the past 120 days, we may have longer period locally for better stock related calculation availability.
