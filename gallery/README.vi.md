# Thư viện HTML tĩnh cho kết quả suy luận

> 🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

Đây là trình dựng trang HTML **bằng CPU** để trình bày output do notebook Qwen3.8-27B Text + Vision tạo ra sau khi thực thi. Người xem có thể so sánh tiếng Anh và tiếng Việt (nếu có kết quả VI), xem 16 ảnh nguồn và SHA256, 19 case ID, EOS, thời gian, lọc nhóm, tìm kiếm và tải JSON bằng chứng. Trang không chạy suy luận.

## Tùy chọn trong notebook Kaggle

[Notebook](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) hoàn thành và lưu **19 case tiếng Anh ban đầu trước**. Các biến môi trường được đọc sau đó:

- **QWEN38_ENABLE_VI=0**: không suy luận thêm; tiếng Việt được đánh dấu rõ *chưa sinh kết quả*. Mỗi thẻ OCR số sử dụng lại kết quả chữ số của chính case đó khi hiển thị VI; hai case OCR hoàn toàn riêng biệt.
- **QWEN38_ENABLE_VI=1** (mặc định): chạy **17 lượt suy luận tiếng Việt mới** (3 Text, 6 Rich Vision, 8 Photography) bằng **cùng MaxEngine đã nạp**, lưu riêng raw text, token IDs, EOS, thời gian và lỗi tại cases/full19-vi.json. Bằng chứng tiếng Anh gốc không bị ghi đè. Gallery được dựng bằng CPU từ JSON kết quả của notebook; output và SHA256 phải gắn với đúng run ID.
- **QWEN38_RENDER_HTML=1** (mặc định): sau suy luận, dùng **CPU** tạo gallery/index.html, gallery/evidence.json, gallery/assets/ và inference-gallery.zip. Đặt 0 nếu không muốn xuất HTML.

Đặt các tùy chọn trước Run All. Python cell đầu tiên có chú thích về giá trị mặc định và có thể sửa nếu cần. Gallery không tự tạo câu trả lời tiếng Việt và không chạy lại OCR.

## Xem Gallery qua Cloudflare Tunnel ngay trong Kaggle

**Cell 17** chỉ mở Cloudflare Quick Tunnel khi Kaggle báo **phiên chỉnh sửa tương tác** (`KAGGLE_KERNEL_RUN_TYPE=Interactive`) và `QWEN38_TUNNEL_ENABLED=1`. Bật **Internet = On** nếu muốn sử dụng trong trình chỉnh sửa Kaggle. URL tạm xuất hiện trong output và chỉ hoạt động khi cell còn chạy.

**Chạy bằng API/CLI và Save & Run All trên UI Kaggle đều là phiên Batch**. Cell 17 tự bỏ qua Cloudflare nếu là Batch hoặc không xác định được môi trường: **không giữ thêm 900 giây, không tải binary từ mạng**. File `inference-gallery.zip` được tạo bằng CPU, evidence JSON và câu trả lời gốc vẫn nằm trong output. Có thể xem/host ZIP trên máy CPU sau mà không phải chạy lại TPU.

Chỉ khi chia sẻ tương tác, `QWEN38_TUNNEL_HOLD_SECONDS=900` là mặc định (cho phép 60–3600 giây), thời gian giữ URL vẫn tính quota phiên TPU. Binary cloudflared 2026.10.5 được kiểm SHA256. Chỉ phục vụ thư mục Gallery; URL hết hạn khi cell/phiên dừng và không lưu thành artifact lâu dài. Nếu mạng chặn, Gallery ZIP vẫn còn. Cell 17 chỉ đổi nhãn HTML từ *Internal review* thành *Inference showcase* và cập nhật ZIP; JSON EN/VI và bằng chứng SHA256 không đổi.

## Dựng lại từ kết quả thật trên CPU

Không cần cài thêm thư viện, kết nối mạng, GPU hay TPU:

    python3 gallery/render.py \
      --en /path/to/cases/full19-final.json \
      --vi /path/to/cases/full19-vi.json \
      --images examples/images \
      --manifest examples/assets_manifest.json \
      --output /path/to/new-gallery

Bỏ --vi nếu chưa chạy tiếng Việt. Bộ dựng yêu cầu JSON tiếng Anh hoàn tất 19 case, kiểm số lượng từng nhóm, SHA256 ảnh và manifest 16 ảnh, đồng thời không ghi đè thư mục output đã có dữ liệu. Nếu SHA256 khác, phải dùng **đúng file ảnh của phiên TPU tương ứng**, không dùng ảnh gần giống hoặc tạo lại.

Tải và giải nén `inference-gallery.zip` trên máy CPU. Thư mục `gallery/` chứa `index.html`, `evidence.json` và `assets/`; có thể xem trang mà không cần clone Git.

Host thư mục đã giải nén hoặc Gallery do renderer vừa tạo:

    python3 -m http.server 8080 --bind 127.0.0.1 --directory /path/to/gallery

Mở http://127.0.0.1:8080 để xem tại máy. Nếu muốn mở Quick Tunnel công khai, trước hết tải/clone repository trên **máy CPU chạy Linux**: ZIP Gallery **không chứa** `gallery/install_cloudflared.sh`. Từ thư mục gốc Git, cài binary (không cần sudo):

    bash gallery/install_cloudflared.sh --check
    bash gallery/install_cloudflared.sh
    "$HOME/.local/bin/cloudflared" --version

Script tải bản Cloudflare 2026.10.5 đã ghim cho Linux x86_64 hoặc arm64, xác minh SHA256 được công bố chính thức, thực hiện chmod 755 và không ghi đè file có sẵn chưa được xác minh. Đặt QWEN38_CLOUDFLARED_BIN_DIR để chọn thư mục cài khác. Với macOS/Windows, tham khảo [hướng dẫn cài đặt của Cloudflare](https://developers.cloudflare.com/tunnel/downloads/).

Trên máy CPU đó, mở terminal khác để tạo link công khai tạm:

    "$HOME/.local/bin/cloudflared" tunnel --url http://127.0.0.1:8080

Nếu đã đổi QWEN38_CLOUDFLARED_BIN_DIR, hãy sửa đường dẫn binary tương ứng. HTTP server phải tiếp tục chạy để tunnel hoạt động.

Dừng phiên TPU sau khi tải ZIP để tiết kiệm quota. Cloudflare Quick Tunnel ngừng hoạt động khi tiến trình cloudflared dừng. Không mở công khai notebook Kaggle hoặc trọng số hạn chế quyền.

## Toàn vẹn dữ liệu và bảo mật

Trang chỉ dùng ảnh tại chỗ, JSON escape chống chèn script và không gọi JS/CSS hay dịch vụ ngoài. Giữ nguyên lỗi và trạng thái chưa hoàn tất. **EOS không bảo đảm câu trả lời đúng**; nội dung cần được đánh giá bằng mắt. Review evidence.json trước khi công khai vì có output và metadata của lượt chạy. Kiểm thử dùng **fixture giả được ghi rõ**, tuyệt đối không công bố chúng như kết quả TPU thật.

[Tái lập](../docs/REPRODUCIBILITY.vi.md) · [Bằng chứng Full19](../docs/FULL19-EVIDENCE.vi.md) · [Phân tích lỗi](../docs/FAILURE-ANALYSIS.vi.md)

## Tiếp tục an toàn mà không mất kết quả đã lưu

Cell VI tùy chọn kiểm tra SHA256 của Full19 gốc rồi tiếp tục các VI case còn thiếu **trong cùng session Kaggle còn chạy**. Không ghi đè câu trả lời đã hoàn thành. Case lỗi/chưa hoàn tất được giữ nguyên; đặt QWEN38_RETRY_VI_INCOMPLETE=1 rồi chạy lại riêng cell VI để thử lại những case đó. Lịch sử lỗi nằm trong retry_history. Prompt được kiểm tra giới hạn prefill/decode trước khi suy luận. Run All lại từ đầu tạo RUN_ROOT mới, không tự khôi phục phiên trước.

Khi đã sinh thêm VI, hãy chạy lại riêng cell Gallery: chương trình chỉ làm mới Gallery thuộc đúng lượt EN gốc đã xác minh và thay thế inference-gallery.zip bằng bản vừa xuất. Hãy tải hoặc sao chép ZIP trước khi refresh nếu muốn giữ bản cũ. Trình dựng độc lập cũng có tùy chọn --refresh, chỉ áp dụng với Gallery thuộc cùng EN run. Dữ liệu VI cần gắn SHA256 nguồn, còn case lỗi phải giữ trạng thái thật, không tạo câu trả lời giả. Quy trình resume/refresh được kiểm thử riêng bằng CPU. Gallery chỉ dựng dữ liệu đầu vào đã được kiểm tra nguồn và SHA256.

## Gallery phân nhóm chức năng

Renderer chạy CPU, phân 19 case thành năm nhóm: biểu đồ/sơ đồ (4), trích xuất dữ liệu ảnh (4), hiểu cảnh (6), OCR chữ số (2) và hỏi đáp văn bản (3). Người xem chọn ví dụ để thấy ảnh nguồn cùng hai câu trả lời EN/VI được suy luận độc lập. Giao diện hoạt động ngay cả khi chưa có file đánh giá thủ công.

Để thêm lớp đối chiếu ngữ nghĩa **tùy chọn và sơ bộ**:

```bash
python3 gallery/render.py \
  --en /path/to/run/cases/full19-final.json \
  --vi /path/to/run/cases/full19-vi.json \
  --images /path/to/verified/images \
  --manifest examples/assets_manifest.json \
  --review /path/to/semantic-review.v1.json \
  --output /path/to/new-gallery
```

Khi không có `--review`, Gallery không tự tạo bản dịch, đáp án tham chiếu hay điểm đánh giá. Notebook gọi renderer từ source bundle đã xác minh và chỉ kiểm tra `RUN_ROOT/semantic-review.v1.json` nếu có. Không thay đổi JSON suy luận và ảnh nguồn.

File review chỉ được chấp nhận khi có đủ 17 case VI hoàn thành, SHA256 EN/VI và SHA256 evidence JSON khớp, đủ 17 ID không thuộc OCR và cấu trúc nội dung hợp lệ. Nhận xét là **đối chiếu thủ công sơ bộ, không phải chứng nhận accuracy hoặc benchmark dịch thuật**. Ba file phụ trợ riêng được xuất khi hợp lệ: `semantic-review.v1.json`, `manual-comparison.en.md`, `manual-comparison.vi.md`.


## Ràng buộc nguồn và refresh an toàn

Mỗi Gallery mới có file `source-full19.sha256` lưu SHA256 của JSON tiếng Anh gốc. Khi refresh, trình dựng kiểm chính xác checksum này, đồng thời đối chiếu metadata và từng case EN. Khi thiếu sidecar, renderer kiểm chi tiết metadata và từng case; không tuyên bố dữ liệu đã được ràng buộc bằng checksum toàn file. Nếu người dùng chỉ định `--vi` nhưng file không tồn tại, chương trình báo lỗi thay vì âm thầm coi là “chưa sinh tiếng Việt”.

Chương trình kiểm tra checksum ảnh, provenance của bản đối chiếu và SHA256 evidence JSON trước khi tạo thư mục đầu ra. File đánh giá không hợp lệ sẽ không để lại thư mục Gallery mới bị tạo dở. Renderer Gallery có thể được kiểm thử trên CPU với dữ liệu giả được ghi nhãn rõ ràng. Việc suy luận TPU và đối chiếu output thuộc lần chạy notebook Kaggle chính thức.


## Một bộ dựng Gallery dùng chung

Nguồn code dựng HTML chuẩn là gallery/render.py và chạy được bằng CPU.
Cell cuối gọi module offline đã xác minh; kiểm tính đồng bộ bằng:

    python3 support/sync_gallery_notebook.py --write
    python3 support/sync_gallery_notebook.py --check

Khi cải tiến HTML, chỉ cần dựng lại từ JSON EN/VI cùng đúng 16 ảnh nguồn,
không cần GPU hoặc TPU. Khi chạy end-to-end, notebook tạo Gallery cùng cấu
trúc từ dữ liệu inference của phiên thực thi đó.

Để đưa ghi chú kiểm tra thủ công vào giao diện, sử dụng tham số
--review path/to/semantic-review.v1.json. Ghi chú phải khớp SHA256 EN, VI,
và dữ liệu Gallery. Chỉ những case có kết quả kiểm tra mới mang nhãn
review; những case khác vẫn hiển thị Chưa đánh giá, không sửa output AI.
