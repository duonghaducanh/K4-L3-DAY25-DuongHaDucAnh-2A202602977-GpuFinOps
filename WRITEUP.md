# Bài viết ngắn — Lab 25: GPU FinOps Optimization

> **Học viên:** Dương Hà Đức Anh · **MSSV:** 2A202602977
> **Lab:** AICB · Phase 2 · Track 2 (Infrastructure) · Day 25
> **Deliverable:** `outputs/report.md`, `outputs/savings.png`, `outputs/focus_export.csv`

---

## 1. Baseline vs. Optimized

NimbusAI đang trả tiền GPU theo hai dòng: **inference** (đo bằng token) và **thuê GPU**
(đo bằng GPU-giờ). Tôi gộp cả hai vào một baseline tháng 6/2026 và đo lại sau tối ưu:

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| Chi phí tháng | **$27,133** | **$13,747** | **−$13,386 (−49%)** |
| Inference `$/1M-token` | $6.488 | $1.126 | **−83%** |
| Inference `$/ngày` | $48.87 | $8.48 | −83% |
| Số token phục vụ/ngày | 7,533,027 | 7,533,027 | không đổi |

Điểm mấu chốt: **sản lượng token không đổi** nhưng chi phí giảm một nửa. Nếu chỉ nhìn
`$/GPU-giờ` ta sẽ không thấy được điều này — con số đó chỉ nói *trả bao nhiêu*, không nói
*nhận được gì*. Chuyển sang `$/1M-token` buộc mọi đòn bẩy phải chứng minh bằng sản lượng.

---

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | Tiết kiệm/tháng | % tổng | Nguồn |
|---|---|---|---|
| Purchasing (spot/reserved) | $10,919 | 82% | M3 |
| Inference (cascade/cache/batch) | $1,212 | 9% | M2 |
| Right-size GPU "util-lie" | $655 | 5% | M1 |
| Kill idle GPUs | $600 | 4% | M1 |

**Purchasing là đòn bẩy lớn nhất (82%).** 8 workload trong `workloads.csv` phần lớn là
pipeline định kỳ có duty cycle cao, nên cam kết reserved/spot ăn chiết khấu 40–45% mà
không cần viết thêm một dòng code nào — đây là "tiền hợp đồng", ROI cao nhất, không rủi ro
kỹ thuật.

**Inference (9%)** là đòn bẩy *kỹ thuật* mạnh nhất: cascade (80% request đủ cho model nhỏ
rẻ 15×) + prompt caching (−90% cho input đã cache) + batch API (−50%) nhân chồng lên nhau.
`discount_stack(batch=True, cache_hit_frac=1.0) = 0.05` — tức **95% off** khi cả hai cùng
áp dụng, vì chiết khấu *nhân* chứ không cộng.

**Right-size + idle (9%)** là các "đồng tiền rơi vãi" — nhỏ nhưng gần như miễn phí để thu.

---

## 3. GPU-Util Lie — tác động tài chính

`gpu-h100-4` báo **GPU-Util 98.2% nhưng MFU chỉ 0.19**. Đây là cái bẫy cốt lõi:

`nvidia-smi` đo **thời gian clock ở trạng thái active**, không đo **FLOPs thực sinh ra**.
Một GPU bị *memory stall* (chờ dữ liệu HBM về), hoặc *kernel-launch overhead* (kernel quá
ngắn, GPU bận dispatch hơn là tính), sẽ đọc ~100% util trong khi tensor core gần như ngồi
chơi. Ta bị tính tiền **cả giờ H100** nhưng chỉ nhận **1/5 FLOPs** đã thuê.

Hệ quả tiền: `gpu-h100-4` lãng phí **~$511/tháng**; thêm `gpu-a10g-1` (util 96.9%, MFU 0.27)
là **$144/tháng**. Tổng **$655/tháng** bị đốt chỉ vì chọn GPU theo "util đẹp" thay vì MFU.

> Bài học: khi mua sắm hạ tầng, **hỏi MFU/MBU, đừng hỏi GPU-Util**. Một con số 98% trông rất
> thuyết phục trên dashboard nhưng có thể là GPU đang bận *chờ*.

---

## 4. Các phần mở rộng đã triển khai (5/5)

### Ext 1 — Purchasing policy v2 (`finops/pricing.py`)
Thay quy tắc "interruptible → spot" bằng hàm **tối thiểu hoá chi phí thực**: (a) giá spot
được *điều chỉnh theo rủi ro* (interruption rate riêng từng GPU: L4 11%/h vs H100 3%/h, cộng
overhead checkpoint + rework), (b) chỉ tính hạn mức reserved mà job *thực sự amortise được*
(1yr vs 3yr).
**Kết quả: M3 savings 39.1% → 42.5% (+3.4 điểm %).** `job-train-llm` (20h/ngày, 8×H100)
chuyển từ spot sang reserved vì spot risk-adjusted đắt hơn reserved. Bảng tier matrix cho
thấy đáp án phụ thuộc *cả* GPU type *lẫn* duty cycle — không có một quy tắc chung đúng cho mọi
job. (Đo độ nhạy: nếu chỉ được cam kết 1 năm, savings tụt còn 31.4%.)

### Ext 2 — Right-sizing theo MBU (`finops/metrics.py`, `missions/m1_efficiency_audit.py`)
Với workload memory-bound (infer/embed), chọn GPU theo **`$/TB/s` băng thông**, không theo
`$/GPU-giờ`. `gpu-a10g-0/1` chỉ cần 0.14–0.18 TB/s nhưng đang thuê A10G ($/TBps = 1.667);
chuyển sang **L4** ($0.80/h, 0.3 TB/s) vẫn dư sức phục vụ → **$288/tháng**. Câu hỏi chấm:
*"Sao không chọn GPU rẻ nhất theo `$/GPU-hr`?"* — vì GPU rẻ nhất có thể **không đủ băng thông**,
khiến decode chậm hơn và phải bù bằng nhiều GPU hơn; `$/TB/s` mới là thước đo đúng.

### Ext 3 — `cache_is_worth_it()` (`finops/pricing.py`, M2)
Cache chỉ có lợi khi prefix được đọc lại đủ nhiều để bù chi phí ghi (write ≈ 1.25× giá input,
read ≈ 0.10×). Tính break-even theo từng tier: **small cần 4.17 lần đọc, large cần 1.57**.
Thực tế dataset đọc lại **476 lần (small)** và **124 lần (large)** — vượt xa ngưỡng, nên khoản
tiết kiệm cache là **thật**, không phải con số "book" cho đẹp. Hàm này được dùng làm *gate*:
nếu break-even không đạt, M2 sẽ không ghi nhận savings cache.

### Ext 4 — Ngân sách Reasoning (M2, M5)
Tách riêng traffic `is_reasoning=1`: **8.4% request, 35.6% output token, 16.5% chi phí $,
nhưng 94% năng lượng (Wh)** — mỗi query reasoning tốn **148.2 Wh vs 0.86 Wh** (~172×). Nghịch
lý: reasoning chỉ chiếm 1/6 hóa đơn tiền nhưng gần như *toàn bộ* dấu chân năng lượng.
**Đề xuất routing rule:** chỉ bật reasoning khi độ phức tạp task vượt ngưỡng (ví dụ cần
multi-step/verification), còn lại route sang model thường — vừa giảm $ vừa cắt mạnh Wh.

### Ext 5 — Carbon-aware Scheduling (`missions/m3_purchasing.py`)
Mọi job `interruptible=1` không có SLO latency với người dùng, nên có thể chạy ở bất kỳ đâu.
Tính `kWh = GPU-hours × TDP`, so `us-east-1` (380 gCO2/kWh, $0.12/kWh) với **europe-north1**
(thủy điện Na Uy, 30 gCO2/kWh, $0.09/kWh). **Kết quả: giảm 626,150 gCO2e (−92%) và tiết kiệm
$54 điện mỗi lần chạy** — vùng sạch nhất *đồng thời* rẻ thứ hai, nên không có trade-off.
Lưu ý: `us-east-wa` rẻ nhất về $ nhưng bẩn hơn europe-north1 (90 vs 30 gCO2/kWh); vùng "tối ưu"
thực sự phụ thuộc công ty ưu tiên $ hay CO2.

---

## 5. Khuyến nghị cho NimbusAI (3 hành động đầu tiên)

1. **Ký cam kết reserved/spot cho các job duty-cycle cao ngay tuần này** — $10,919/tháng,
   không cần engineering, ROI cao nhất, làm trước tiên. Dùng policy v2 để chọn tier theo
   *risk-adjusted cost*, không theo rule-of-thumb.
2. **Đo MFU/MBU thay vì GPU-Util, và right-size các GPU "util-lie"** — thu hồi $655/tháng
   đang bị đốt, đồng thời bật auto-shutdown cho GPU idle (thêm $600/tháng). Đây là thay đổi
   monitoring + scheduler, không phải mua sắm.
3. **Đưa job interruptible sang europe-north1 và đặt routing rule cho reasoning** — cắt 92%
   carbon của khối training + giới hạn reasoning (94% năng lượng) chỉ khi task thật sự cần.
   Bước này biến FinOps từ "tiết kiệm tiền" thành "tiết kiệm tiền *và* carbon".

**Tổng kết:** giảm **49% chi phí** ($27,133 → $13,747) và **83% `$/1M-token`** trong khi giữ
nguyên sản lượng token — phần lớn đến từ hợp đồng mua GPU, phần còn lại từ kỷ luật đo lường
đúng chỉ số (MFU/MBU thay vì GPU-Util) và các đòn bẩy inference xếp chồng.

---

*Số liệu là snapshot tháng 6/2026, sinh tất định bằng `data/generate.py` (seed=25); cần
re-baseline trước khi áp dụng thực tế. Toàn bộ code chạy được trên laptop không GPU.*
