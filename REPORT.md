# Báo cáo Phân tích Benchmark Memory Systems cho AI Agent

Dựa trên kết quả chạy benchmark của `BaselineAgent` và `AdvancedAgent` thông qua hai kịch bản `Standard Benchmark` và `Long Context Stress Benchmark`, tôi xin đưa ra phân tích chi tiết nhằm trả lời 4 câu hỏi trọng tâm của bài thực hành.

## 1. Vì sao Advanced Agent có Recall tốt hơn Baseline Agent qua nhiều session?
Trong bảng kết quả (cột `Cross-session recall`), Baseline Agent cho kết quả = 0, trong khi Advanced Agent đạt 1 (hoàn hảo).
Lý do nằm ở cách thiết kế bộ nhớ dài hạn (Persistent Memory). Baseline Agent chỉ lưu giữ các `messages` trong bộ nhớ ngắn hạn của cùng một `thread_id` (session). Khi người dùng hỏi một thông tin cá nhân ở một thread hoàn toàn mới (`thread_2`), bộ nhớ của Baseline bị reset trắng, do đó nó không thể trả lời. Ngược lại, Advanced Agent sử dụng `UserProfileStore` để lưu vĩnh viễn các "facts" (ví dụ: Tên, Sở thích, Nơi ở) ra một file vật lý (`User.md`). Nhờ vậy, dù đổi sang thread mới, hệ thống vẫn đọc lại thông tin từ `User.md` và cung cấp chính xác cho LLM.

## 2. Vì sao ở hội thoại ngắn, Advanced Agent lại tiêu thụ nhiều token prompt hơn Baseline?
Tại `Standard Benchmark`, `Prompt tokens processed` của Advanced Agent (22000) lớn hơn một chút so với Baseline (21931).
Nguyên nhân là do mỗi lần phản hồi, Advanced Agent luôn phải trích xuất và nối thêm toàn bộ nội dung của file `User.md` (chứa Profile của người dùng) vào trong `System Prompt` để đảm bảo không bị quên thông tin. Trong các cuộc hội thoại ngắn, khi số lượng tin nhắn chưa nhiều, phần chênh lệch do việc "đính kèm" file Profile này làm cho tổng số lượng prompt token của Advanced Agent lớn hơn Baseline Agent một lượng nhỏ.

## 3. Vì sao cơ chế Compact Memory tối ưu được ở hội thoại siêu dài?
Tại `Long Context Stress Benchmark`, chúng ta chứng kiến một sự đảo ngược: `Prompt tokens processed` của Baseline (21931 - hoặc có thể cao hơn nhiều nếu không tính offlline mode) bắt đầu vượt xa Advanced Agent (17172). 
Đó là nhờ `CompactMemoryManager`. Khi số lượng token vượt quá ngưỡng `threshold_tokens` (ví dụ 2000 tokens), Advanced Agent sẽ tự động tóm tắt các tin nhắn cũ (`summarize`) và chỉ giữ lại một vài tin nhắn gần nhất (`keep_messages`). Việc này biến một lịch sử chat phình to vô hạn thành một chuỗi context có độ dài ổn định. Nhờ đó, lượng token đẩy lên API (Prompt tokens) được kìm hãm, giảm thiểu đáng kể chi phí gọi LLM so với cơ chế lưu full history của Baseline.

## 4. Tăng trưởng bộ nhớ (Memory file) diễn ra như thế nào và rủi ro kèm theo
Trong các bài benchmark, `Memory growth (bytes)` của Advanced Agent tăng nhẹ (từ 0 lên 33-35 bytes) do nó tạo ra và ghi vào file `User.md`.
**Cách tăng trưởng:** Mỗi khi người dùng cung cấp một fact mới (ví dụ "Tôi tên là...", "Tôi sống ở..."), hàm `extract_profile_updates` sẽ lấy thông tin đó và nối/cập nhật vào file.
**Rủi ro đi kèm trong thực tế:**
- **Phình to bộ nhớ (Bloating):** Nếu người dùng chat quá lâu, cung cấp vô số thông tin không quan trọng, file `User.md` sẽ bị phình to làm tăng `prompt tokens` cho mỗi request.
- **Xung đột / Sai lệch sự thật (Conflicting Facts):** Người dùng có thể thay đổi thông tin (hôm qua nói ở HN, hôm nay chuyển vào SG). Nếu thiết kế tồi, file sẽ lưu cả 2, khiến LLM bối rối.
> **Note (Bonus):** Rủi ro xung đột thông tin này đã được giải quyết bằng tính năng **Conflict Handling** (Ghi đè thông tin dựa trên Key như Name, Location thay vì nối thẳng vào cuối file), giúp file `User.md` luôn nhất quán!
