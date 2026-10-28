# Kiểm tra Git history và những nhánh backup cũ

> [English](GIT-HISTORY-AUDIT.md) | **Tiếng Việt**

## Nhánh công bố và nhánh chuẩn bị cục bộ

Xác minh ngày 28/10/2026: main trên GitHub vẫn là một root commit nhập snapshot bcac36a, ngày 19/10. Subject tập trung vào README nên không mô tả đủ **82 file và 3.229 dòng thêm**, gồm notebook, API, tests và assets. Giờ author và committer lệch khoảng 3 giờ 10 phút. Không diễn giải snapshot này thành một loạt commit tính năng chưa được xác minh.

Nhánh LOCAL prepublication-review-20261027 được khôi phục nguyên vẹn, với **bốn commit mới** có body rõ và author/committer date trùng nhau:

| SHA | Nội dung | UTC |
|---|---|---|
| 6ac67f8 | Bổ sung năm ảnh, đủ 16 ảnh và Photography cases | 27/10 16:47:16 |
| 3f93c38 | Ghi nhận Full19 experimental thật và các giới hạn | 27/10 16:47:16 |
| 3804889 | Notebook Split-Vision 320/768 duy nhất và kiểm tra nguồn | 27/10 17:04:25 |
| 571145a | Hướng dẫn offline support và release gates | 27/10 17:04:25 |

Các commit có cùng giây được giữ đúng lịch sử lưu thực tế, **không chỉnh lại ngày giả**. README EN/VI vẫn là bản nháp chưa commit cho đến khi hoàn tất nghiệm thu.

## Audit backup remote

GitHub Private hiện lưu tám nhánh backup. Một số nhánh:

| Hậu tố tên backup | Số commit |
|---|---:|
| pre-feature-history-rewrite-20261019 | 28 |
| pre-message-readme-rewrite-20261019 | 28 |
| pre-evidence-timestamp-rewrite-20261019 | 14 |
| pre-governance-rewrite-20261019 | 11 |

Đây là các ứng viên backup/rewrite trước đó, **không phải Git history hiện tại của main**. Ứng viên 28 commit khác main ở **27 file**, gồm notebook đã chạy cũ, evidence cũ, các phiên bản ảnh và README. So sánh cây từ backup sang main có khoảng **387 dòng thêm và 4.243 dòng xóa**. Reset main thẳng về backup sẽ làm tài liệu cũ xuất hiện lại.

## Khuyến nghị trước công bố

Giữ nguyên snapshot nhập ban đầu cùng các commit mới có phạm vi rõ. Nếu muốn migration lịch sử, phải duyệt phương án riêng, chứng minh chính xác Git tree cuối, SHA256, backup và chỉ cập nhật non-fast-forward sau khi chấp thuận. Trước khi set Public cần kiểm các nhánh backup/review vì chúng có thể làm lộ evidence cũ.

Không push, merge, xóa hoặc rewrite remote trong đợt audit này. Fresh TPU Run All chính thức vẫn chưa diễn ra.
