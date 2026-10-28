# Ghi công thư viện và dự án nguồn mở

> 🌐 Ngôn ngữ / Language: [English](THIRD_PARTY_NOTICES.md) | **Tiếng Việt**

Bản demo TPU sử dụng nhiều dự án nguồn mở. Phần mềm và model vẫn chịu giấy phép riêng, độc lập với [MIT của mã nguồn dự án](LICENSE).

| Dự án | Vai trò | Nguồn và giấy phép |
|---|---|---|
| [bản mirror BF16 trên Kaggle](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) (nguồn Qwen upstream) | Checkpoint BF16 dành cho suy luận trên Kaggle TPU | [Bản sao giấy phép Apache-2.0 upstream](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0) |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext) | Runtime transformer cho TPU | Kiểm revision và giấy phép đúng phiên bản |
| [JAX](https://github.com/jax-ml/jax) | Biên dịch và tính toán TPU | Apache-2.0 upstream |
| [FastAPI](https://github.com/fastapi/fastapi) | Demo HTTP API | MIT upstream |
| [Vue](https://github.com/vuejs/core) | Giao diện trình duyệt | MIT upstream |
| [Playwright](https://github.com/microsoft/playwright) | E2E browser tests | Giấy phép theo upstream |

Dataset support còn có Python wheels, helpers, patches và cache biên dịch. Các thành phần này **không phải đều do dự án tự viết**; phải kiểm version, license và quyền phân phối trước khi đóng gói lại.

Xem [nguồn gốc model](MODEL_LICENSE.vi.md) để phân biệt mã dự án và pretrained weights. URL hay hash chỉ chứng minh nguồn tham chiếu, **không mặc nhiên cấp quyền phân phối mọi artifact**.

## Ghi nhận khi phân phối Runtime Support

Archive MaxText đã điều chỉnh có thông báo sửa đổi trong các file tích hợp và bản tổng hợp thay đổi; giấy phép Apache-2.0 upstream được giữ nguyên. Wheel xprof 2.23.1 khai báo Apache-2.0 nhưng chưa tìm thấy file license đính kèm được nhận diện riêng trong lần kiểm tra này. Runtime Support Dataset cung cấp bản sao Apache-2.0 độc lập. Các wheel khác vẫn có điều khoản giấy phép upstream riêng.


### Giấy phép upstream xprof 2.23.1 (đã rà soát)

Nguyên bản [LICENSE của xprof-v2.23.1](https://github.com/openxla/xprof/blob/xprof-v2.23.1/LICENSE) được lưu tại [XPROF-2.23.1-LICENSE](THIRD_PARTY_LICENSES/XPROF-2.23.1-LICENSE). Nội dung gồm Apache-2.0, copyright TensorFlow Authors và điều khoản BSD của Polymer/Web Components. Wheel không chứa LICENSE/NOTICE độc lập; tag nguồn không có NOTICE ở thư mục gốc. Các file JS đóng gói còn chứa Angular MIT và copyright Chromium, được giữ nguyên trong wheel; không nên mô tả toàn bộ các tài nguyên này chỉ có Apache-2.0.

### Ghi nhận giấy phép frontend đóng gói (đã kiểm tra)
Các JavaScript bundle có header MIT của Angular 15.2.9 (zone.js còn ghi Angular v15.1.0-next.0). Bản license MIT đúng tag Angular 15.2.9 được giữ tại [ANGULAR-15.2.9-MIT-LICENSE](THIRD_PARTY_LICENSES/ANGULAR-15.2.9-MIT-LICENSE).

Lit Web Components có SPDX BSD-3-Clause trong bundle; bản license upstream tại [LIT-BSD-3-CLAUSE-LICENSE](THIRD_PARTY_LICENSES/LIT-BSD-3-CLAUSE-LICENSE). Không xác định được chính xác version Lit từ bundle.

Tracing viewer có copyright và BSD của Chromium; giữ toàn bộ [CHROMIUM-BSD-3-CLAUSE-LICENSE](THIRD_PARTY_LICENSES/CHROMIUM-BSD-3-CLAUSE-LICENSE). Điều khoản 0BSD của Microsoft tslib đã có trong JS, bổ sung bản nguyên văn tại [TSLIB-0BSD-LICENSE](THIRD_PARTY_LICENSES/TSLIB-0BSD-LICENSE).

Material Icons WOFF2 khớp byte với đúng asset ở tag XProf 2.23.1. Google công bố Material Icons theo Apache-2.0; Dataset đã có bản license tương ứng. Không tách, sửa hay phát hành riêng file font ngoài wheel upstream.

Nguồn: https://github.com/angular/angular/blob/15.2.9/LICENSE ; https://github.com/lit/lit/blob/main/LICENSE ; https://github.com/chromium/chromium/blob/main/LICENSE ; https://github.com/microsoft/tslib/blob/main/LICENSE.txt ; https://developers.google.com/fonts/docs/material_icons .

Đây là danh mục giấy phép phát hiện từ wheel **không phải kiểm kê đầy đủ toàn bộ chuỗi phụ thuộc**. Chưa xác minh hết thư viện được liên kết bên trong mã native/WASM.

### Giấy phép nguồn build WASM (giới hạn phạm vi)

[XProf frontend BUILD](https://github.com/openxla/xprof/blob/xprof-v2.23.1/frontend/app/components/trace_viewer_v2/BUILD) và [WORKSPACE](https://github.com/openxla/xprof/blob/xprof-v2.23.1/WORKSPACE) của tag 2.23.1 ghim các phụ thuộc phục vụ WebAssembly: **Dear ImGui 1.88** (MIT), **Perfetto 53.0** (Apache-2.0 và điều khoản kèm theo), **Dawn WebGPU v20250713.025201** (BSD), **Emscripten 4.0.6** (MIT/University of Illinois-NCSA). Các bản license upstream được giữ nguyên tại [IMGUI-1.88-MIT-LICENSE](THIRD_PARTY_LICENSES/IMGUI-1.88-MIT-LICENSE), [PERFETTO-53.0-LICENSE](THIRD_PARTY_LICENSES/PERFETTO-53.0-LICENSE), [DAWN-20250713-BSD-LICENSE](THIRD_PARTY_LICENSES/DAWN-20250713-BSD-LICENSE) và [EMSCRIPTEN-4.0.6-LICENSE](THIRD_PARTY_LICENSES/EMSCRIPTEN-4.0.6-LICENSE). Dependency build cũng có **Abseil C++** theo Apache-2.0.

Đây là các dependency được xác định từ source build; chưa tái dựng và xác minh đầy đủ danh sách thư viện liên kết vào binary WASM/native. Việc kèm license **không đồng nghĩa kiểm toán chuỗi cung ứng đã hoàn tất**.
