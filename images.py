import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

# Tạo hình ảnh minh họa
fig, ax = plt.subplots(figsize=(10, 5))

# Chèn hình ảnh cô dâu
img = plt.imread('images/testbride.jpeg')  # Đường dẫn ảnh của người dùng
ax.imshow(img, extent=[0, 2, 2, 5])

# Vẽ khối CNN
cnn_box = plt.Rectangle((3, 2.5), 2, 2, color='lightgreen', alpha=0.7)
ax.add_patch(cnn_box)
ax.text(4, 3.3, "CNN\nfor image\nclassification", ha='center', va='center', fontsize=12)

# Mũi tên từ ảnh đến CNN
arrow1 = FancyArrow(2, 3.5, 0.8, 0, width=0.1, head_width=0.3, color='black')
ax.add_patch(arrow1)

# Mũi tên từ CNN đến kết quả
arrow2 = FancyArrow(5, 3.5, 0.8, 0, width=0.1, head_width=0.3, color='black')
ax.add_patch(arrow2)

# Vẽ kết quả dự đoán
ax.text(6.5, 3.8, "Predicted Classes", fontsize=12, weight='bold')
classes = ["Clothing", "Event", "Costume", "Red", "Performance art"]
for i, cls in enumerate(classes):
    ax.text(6.5, 3.5 - i * 0.5, cls, fontsize=10)
ax.text(7, 3.5 - 0.5, "X", fontsize=20, color='red', weight='bold')  # Dấu X màu đỏ

# Tắt trục
ax.axis('off')
plt.show()
