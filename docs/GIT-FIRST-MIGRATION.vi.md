# Notebook Kaggle TPU Git-first — hướng dẫn chuyển đổi

[English](GIT-FIRST-MIGRATION.md)

## Những thay đổi

Notebook **duy nhất** tại [notebooks/kaggle-tpu-v5e8-production-showcase.ipynb](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) lấy Python source trực tiếp từ GitHub. Notebook có 17 cell code: 16 bước workflow tuần tự (19 case EN, tùy chọn 17 lượt sinh VI và HTML Gallery bằng CPU) và một cell tùy chọn mở Gallery Tunnel tương tác. Phiên chạy Kaggle offline cũ vẫn là bằng chứng lịch sử, không tạo thêm notebook thứ hai trong Git.

**Runtime Assets Dataset mới** không còn đóng gói Python implementation của dự án. Bảy gói hỗ trợ vẫn chứa **48 payload** nguyên bản, bao gồm wheel, ảnh đầu vào và **source MaxText bên thứ ba** đã ghim checksum. MaxText là dependency runtime, **không phải** 23 module Python của dự án.

## Thiết lập Kaggle

1. Bật **Internet = On** trước khi chạy cell đầu tiên, để clone GitHub.
2. Attach model Qwen3.8-27B BF16 gốc trên Kaggle (Transformers/default/version 1).
3. Attach [Runtime Assets Dataset v1](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-assets) (đã sẵn sàng, private). Kaggle cung cấp bảy archive runtime và 48 payload SHA256 đã xác minh, không chứa source bundle của dự án. Dataset v1 lịch sử được giữ nguyên.
4. GitHub hiện public và cho phép clone ẩn danh, không cần thông tin xác thực GitHub. Model BF16 và Runtime Assets Dataset là hai Input riêng: phải bảo đảm người dùng Kaggle có quyền truy cập trước khi tuyên bố có thể tái lập từ tài khoản khác.
5. Notebook mặc định dùng `main` và ghi Git HEAD thực tế của mỗi lượt chạy. Khi chủ động tái lập một lượt chạy lịch sử, có thể đặt `QWEN38_GIT_REF` bằng **commit SHA 40 ký tự** trước bootstrap. `QWEN38_EXPECTED_COMMIT` cũng có thể đối chiếu SHA khi clone nhánh/tag; sai SHA sẽ dừng **trước khi khởi tạo TPU**.
6. Chạy tuần tự các cell. Bước 1 xác minh SHA256 48 file runtime và kiểm Git HEAD, worktree sạch, origin URL; mỗi run ghi `source-provenance.json` cùng các kết quả TPU.

Nhánh `main` **không cố định** vì dùng khi phát triển. Không gọi đây là benchmark đã ghim phiên bản. Mỗi run vẫn ghi lại chính xác commit thực thi.

## Dành cho người bảo trì: cách đóng gói Runtime Assets Dataset v1 đã phát hành

Bộ đóng gói kiểm tra đầy đủ bảy archive và 48 payload từ Dataset cũ, giữ nguyên byte của manifest, giấy phép và thông báo bên thứ ba, nhưng **không đưa source ZIP và manifest Python code**:

```bash
python3 support/package_runtime_assets.py \
  --source /path/to/verified-legacy-dataset-v1 \
  --destination /path/to/new-empty-runtime-assets
```

Nguồn có thể là bảy ZIP nguyên bản hoặc các thư mục Kaggle đã giải nén, kèm manifest và notices. Dataset ID đề xuất:

`dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-assets`

**Runtime Assets Dataset private version 1** đã sẵn sàng. Tất cả 48 payload SHA256 đã được kiểm tra độc lập sau khi tải lại Dataset từ Kaggle. Thay đổi Git source không yêu cầu đóng gói lại Dataset.

## Nghiệm thu và giới hạn

```bash
python3 support/git_first_bootstrap.py --check
python3 -m pytest -q tests
npm run test:e2e
```

CPU tests bao gồm clone Git nội bộ với SHA cố định, từ chối SHA sai, đóng gói Dataset mới và kiểm tra regression của notebook cũ. Chúng **không thay thế** việc chạy notebook Git-first thật trên Kaggle TPU. Cần nghiệm thu TPU mới trước khi tuyên bố luồng Git-first đạt production.

Không chỉnh câu trả lời TPU cũ, không gán Git SHA mới cho lần chạy TPU lịch sử dùng embedded source, và không thay đổi Dataset v1.
