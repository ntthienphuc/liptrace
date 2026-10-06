# Định vị LipTrace cho bài software

Tên gợi ý: **LipTrace: A traceable character-CTC lip-reading toolkit with dataset
contracts and stage-level runtime verification**.

Phần đóng góp có thể viết ngay từ code: (1) audit dữ liệu gắn với điều kiện CTC
và vai trò train/val/test; (2) bundle khóa weights với ordered charset, blank và
preprocessing; (3) replay tìm bước bắt đầu khác, bao gồm quyết định rời rạc khi
logits vẫn nằm trong tolerance; (4) workflow train → predict → export có demo
tái lập không phụ thuộc dữ liệu riêng.

Model CNN–BiGRU, CTC, ONNX và hashing đều có prior art. Trình bày đóng góp là
phần mềm tích hợp và xử lý các failure mode cụ thể. Không dùng “first”, “novel
LipNet architecture”, “state of the art”, “Jetson validated” hay “reliable real-time”
khi chưa có thí nghiệm tương ứng.

Kết quả phù hợp để report: số case lỗi đã thiết kế và phát hiện, false rejection
trên healthy fixtures, bước lệch được chẩn đoán đúng, tensor/logit error sau
export cùng CTC/transcript agreement, khả năng cài wheel trên các hệ điều hành,
thời gian/cấu hình môi trường của quy trình kiểm chứng. Tách các kết quả này khỏi
CER/WER trên corpus người thật; fixture tổng hợp không chứng minh accuracy.

Để mạnh hơn cho bài: dùng một subset video thật có quyền sử dụng, lưu danh sách
và hash, kiểm chứng replay PyTorch/ONNX trên nhiều clip/độ dài, mô tả source-group
thực tế và ghi nhận effort tái sử dụng của một người dùng độc lập. Jetson Nano
có thể trở thành một case study sau khi cài được runtime tương thích và đo trên
máy thật; repo v0.1.0 chưa có kết quả đó. Các phần này không cần được dựng lên
để public source, nhưng sẽ nâng chất lượng bằng chứng cho bài.
