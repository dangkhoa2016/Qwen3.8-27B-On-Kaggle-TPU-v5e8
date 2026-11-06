# Qwen3.8-27B BF16 trên Kaggle TPU v5e-8

**BF16 nguyên gốc · Text + Vision · JAX / MaxText · Ví dụ kỹ thuật có thể tái lập**

🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

[![CPU checks](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml/badge.svg)](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml)
[![Mã dự án: MIT](https://img.shields.io/badge/Project_code-MIT-green)](LICENSE)
[![Model: Apache 2.0](https://img.shields.io/badge/Upstream_model-Apache--2.0-blue)](MODEL_LICENSE.vi.md)
![Kaggle TPU v5e-8](https://img.shields.io/badge/Accelerator-TPU_v5e--8-blue)
![BF16 gốc](https://img.shields.io/badge/Checkpoint-Original_BF16-purple)

Repository cung cấp **notebook template Text + Vision có thể thực thi**
cho Qwen3.8-27B BF16 với JAX/MaxText trên Kaggle TPU v5e-8.
Dự án có dữ liệu offline được ghim bằng SHA256, kế hoạch đánh giá Full19,
suy luận tiếng Việt tùy chọn, HTML Gallery xuất bằng CPU
và tài liệu song ngữ EN/VI.

**Chạy và đánh giá:** import notebook vào Kaggle, attach Model và Dataset,
chạy các cell trên TPU v5e-8 rồi xem JSON, token IDs, EOS, thời gian
và HTML Gallery. Kiểm thử CPU kiểm mã nguồn và UI; model trả lời khi chạy trên Kaggle.

## Ảnh nguồn và phạm vi thử nghiệm

[Xem 16 ảnh được quản lý cùng SHA256](examples/README.vi.md).
Full19 gồm **3 Text, 2 OCR số, 6 Rich Vision và 8 Photography**.
Một số ảnh đầu vào là minh họa tạo bằng AI.
Đây không phải ảnh notebook sinh.

Notebook hỗ trợ thêm 17 câu trả lời tiếng Việt và **mặc định bật**.
Câu hỏi vẫn bằng tiếng Anh, được nối thêm chỉ dẫn chỉ trả lời tiếng Việt; đây **không phải prompt gốc bằng tiếng Việt**.
Đặt QWEN38_ENABLE_VI=0 trước Run All nếu muốn bỏ qua.
Mỗi thẻ OCR số dùng lại kết quả của chính case đó khi hiển thị VI;
không chạy thêm lần suy luận OCR tiếng Việt.

### Xem trước các ảnh đầu vào trong Git

Đây là **ảnh nguồn đưa vào model**, không phải kết quả model sinh ra. Bộ ví dụ có ảnh minh họa tạo bằng AI.

| Vạch sang đường | Trạm robot minh họa | Bữa ăn Việt Nam |
|---|---|---|
| [![Vạch sang đường](examples/previews/photo_ai_city_crossing_preview.jpg)](examples/images/photo_ai_city_crossing_1280x960.png) | [![Trạm robot minh họa](examples/previews/vision_3d_robotics_station_preview.jpg)](examples/images/vision_3d_robotics_station.jpg) | [![Bữa ăn Việt Nam](examples/previews/photo_ai_vietnamese_meal_preview.jpg)](examples/images/photo_ai_vietnamese_meal_1280x960.png) |
| Cảnh và quan hệ không gian | Vật thể và bố cục | Nhận diện món ăn |

[Xem đủ 16 ảnh, prompt và dữ liệu SHA256](examples/README.vi.md).

## Chạy notebook trên Kaggle

1. [Mở notebook template](notebooks/kaggle-tpu-v5e8-production-showcase.ipynb)
   trên GitHub, chọn **Raw / Download**, lưu `.ipynb` rồi import vào Kaggle.
2. Chọn **TPU v5e-8** với tám thiết bị TPU.
3. Attach [Kaggle BF16 Model](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b)
   tại Transformers / default / version 1.
4. Attach [Runtime Support Dataset v1](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support).
   Kaggle có thư mục `notebook_source/` (23 module) thay cho `notebook_source.zip`;
   `notebook-source-manifest.json` phải khớp SHA256 ghim trong notebook.
   Dùng version 1; Model BF16 phải attach riêng.
5. Run All trong một phiên TPU. Preflight kiểm tra checkpoint,
   module Python có SHA256 và 48 file dữ liệu hỗ trợ hiện có.
6. Kiểm tra và tải raw text, token IDs, EOS, lỗi, thời gian từng case
   cùng các file Gallery tại /kaggle/working.
7. **Chỉ trong phiên tương tác ở trình chỉnh sửa Kaggle:** bật **Internet = On** và mở link Cloudflare từ cell 17 khi còn chạy (tối đa 15 phút). **API/CLI và Save & Run All dạng batch tự bỏ qua Tunnel và thời gian giữ TPU 15 phút.** Mọi phiên vẫn xuất `inference-gallery.zip` ở cell 16.

## Cấu hình notebook

Cell Python đầu cho phép sửa mặc định, gồm `DEFAULT_TUNNEL_ENABLED` và `DEFAULT_TUNNEL_HOLD_SECONDS`. Biến môi trường đã đặt trước cell này được ưu tiên. Luôn giữ kiểm tra SHA256 và phiên bản runtime.

| Biến môi trường | Mặc định | Chức năng |
|---|---|---|
| `QWEN38_ENABLE_VI` | `1` | Sinh thêm 17 câu trả lời tiếng Việt; `0` để bỏ qua. |
| `QWEN38_RETRY_VI_INCOMPLETE` | `0` | Thử lại case VI chưa hoàn tất chỉ khi chạy lại cell VI; `1` để bật. |
| `QWEN38_RENDER_HTML` | `1` | Dựng HTML Gallery bằng CPU; `0` để bỏ qua. |
| `QWEN38_TUNNEL_ENABLED` | `1` | Cho phép Tunnel **chỉ ở phiên tương tác Kaggle**; batch/API luôn bỏ qua, `0` còn tắt chia sẻ tương tác. |
| `QWEN38_TUNNEL_HOLD_SECONDS` | `900` | Giữ URL chỉ trong phiên tương tác từ 60–3600 giây; batch/API không giữ TPU. |
| `QWEN38_SUPPORT_DIR` | auto | Ghi đè đường dẫn mount của Dataset hỗ trợ. |
| `QWEN38_MODEL_DIR` | auto | Ghi đè đường dẫn mount của Model BF16. |
| `QWEN38_ALLOW_PINNED_WEB_WHEELS` | `0` | Cho phép tải wheel đúng phiên bản khi Kaggle bật Internet; `1` để bật. |

**Tùy chỉnh đường dẫn mount:** Nếu Kaggle không tìm thấy input đã attach, thêm một code cell **trước** cell bootstrap. Thay các đường dẫn ví dụ bằng đường dẫn hiển thị trong **Add Input**:

```python
import os
os.environ["QWEN38_SUPPORT_DIR"] = "/kaggle/input/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support"
os.environ["QWEN38_MODEL_DIR"] = "/kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b/transformers/default/1"
```

Hai đường dẫn này chỉ là tùy chọn. Giữ nguyên cell bootstrap có xác minh SHA256 nguồn.

19 case tiếng Anh gốc được chạy trước. Kiểm phiên bản Dataset attach và SHA256 nguồn tại preflight; badge CI CPU thành công không chứng minh suy luận TPU.

Cell 16 dựng HTML Gallery tĩnh **bằng CPU** từ kết quả notebook vừa sinh, không gọi model thêm lần nữa. Cell 17 tùy chọn mở Cloudflare Quick Tunnel tạm thời, chỉ phục vụ thư mục Gallery và chỉ cập nhật nhãn trình bày HTML. URL hết hiệu lực khi phiên dừng. Xem [hướng dẫn Gallery](gallery/README.vi.md).

## Tài liệu

- [Hướng dẫn thực thi Kaggle](docs/REPRODUCIBILITY.vi.md)
- [Kiến trúc](docs/ARCHITECTURE.vi.md)
- [Các bước notebook ngắn gọn và cách đóng gói source offline](docs/NOTEBOOK-ARCHITECTURE.vi.md)
- [Danh mục Full19 và checklist bằng chứng](docs/FULL19-EVIDENCE.vi.md)
- [Tiêu chí review số liệu và tài liệu ảnh](docs/FAILURE-ANALYSIS.vi.md)
- [Phương pháp đo hiệu năng](docs/PERFORMANCE-ANALYSIS.vi.md)
- [Checklist nghiệm thu](docs/ACCEPTANCE.vi.md)
- [Runtime Support offline](docs/OFFLINE-SUPPORT.vi.md)

## Kiểm thử source không sử dụng TPU

Chạy các lệnh sau mà không tiêu tốn quota TPU:

    python -m pip install -r requirements-cpu.txt
    python -m pytest -q tests
    npm ci
    npx playwright install chromium
    npm run test:e2e
    python3 support/sync_notebook_bootstrap.py --check
    python3 support/sync_vi_state_notebook.py --check
    python3 support/sync_gallery_notebook.py --check

Các lệnh này kiểm tra code và trình duyệt. **Chất lượng câu trả lời và
thông lượng TPU phải được xác minh sau khi chạy notebook chính thức.**

## Cấu trúc dự án

- **notebooks/** — notebook BF16 Text + Vision ngắn gọn cho TPU
- **qwen38_runtime/** — các bước thực thi TPU tách module để kiểm thử
- **examples/** — ảnh đầu vào, prompt và SHA256
- **support/** — công cụ đóng gói và kiểm tra manifest
- **gallery/** — mã dựng HTML bằng CPU và hướng dẫn song ngữ
- **api/** và **web/** — API và playground tùy chọn
- **tests/** — kiểm thử Python và trình duyệt
- **docs/** — kiến trúc, hướng dẫn thực thi và phương pháp đánh giá

## Giấy phép và đóng góp

Code và tài liệu của dự án dùng [MIT](LICENSE).
Checkpoint Qwen và các thành phần bên thứ ba có giấy phép riêng:
[giấy phép model](MODEL_LICENSE.vi.md) và [thông báo bên thứ ba](THIRD_PARTY_NOTICES.vi.md).
Git không chứa trọng số model. Đây là dự án kỹ thuật độc lập,
không được Qwen chứng thực hay bảo trợ.

[Đóng góp](CONTRIBUTING.vi.md) · [Hỗ trợ](SUPPORT.vi.md) ·
[Bảo mật](SECURITY.vi.md) · [Nhật ký thay đổi](CHANGELOG.vi.md)
