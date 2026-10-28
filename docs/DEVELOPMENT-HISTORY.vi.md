# Lịch sử phát triển repository

> [English](DEVELOPMENT-HISTORY.md) | **Tiếng Việt**

Git repository hiện bắt đầu bằng **một root commit** ngày 19/10/2026 (`bcac36a`). Commit này chứa chung source, notebook, tests, ảnh và README tiếng Anh/Việt. Không được coi một commit tổng hợp là bằng chứng cho nhiều mốc commit kỹ thuật trong quá khứ.

Các thay đổi tiếp theo cần chia thành commit đúng phạm vi, có author/committer timestamp thực tế và phần body giải thích rõ. Không sửa ngày để giả lập quá trình phát triển. Khi phân tách snapshot phải giữ nguyên các file gốc và tính tái lập. Cặp README Anh/Việt nên là commit **cuối cùng sau khi** code, notebook, tests và evidence/docs đã hoàn thiện.

Chỉ tái cấu trúc root commit thành các commit theo chức năng khi đã duyệt một kế hoạch rewrite riêng, kiểm chứng được bản backup và đồng bộ ref remote. Không force-push lên `main` hoặc nói đã có backup branch nếu chưa kiểm chứng.

Xem [nghiệm thu](ACCEPTANCE.vi.md), [tái lập](REPRODUCIBILITY.vi.md), [bằng chứng Full19](FULL19-EVIDENCE.vi.md).

Các nhánh backup remote đã được xác minh; xem [báo cáo Git history](GIT-HISTORY-AUDIT.vi.md) để biết khác biệt và rủi ro khi chuyển Public.
