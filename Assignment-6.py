import numpy as np
import matplotlib.pyplot as plt

M = float(input("Enter a value for M: "))

x = np.linspace(-10, 10, 500)

y1 = M * x**2
y2 = M * np.sin(x)

plt.plot(x, y1, label=r'$y = M \cdot x^2$', color='blue', linestyle='--')
plt.plot(x, y2, label=r'$y = M \cdot \sin(x)$', color='red', linestyle='-')

plt.legend()
plt.grid(True)
plt.title('Mathematical Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#2nd question
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Subject': ['Math', 'Physics', 'Chemistry', 'Biology', 'English'],
    'Score': [85, 90, 78, 88, 92]
}

df = pd.DataFrame(data)

plt.figure(figsize=(8, 5))
sns.barplot(x='Subject', y='Score', data=df, palette='viridis')

for index, row in df.iterrows():
    plt.text(index, row['Score'] + 1, row['Score'], ha='center', va='bottom')

plt.title('Scores by Subject')
plt.xlabel('Subjects')
plt.ylabel('Scores')
plt.grid(axis='y')
plt.show()

#3rd question
import numpy as np
import matplotlib.pyplot as plt

seed = 12345
np.random.seed(seed)

data = np.random.randn(50)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(np.cumsum(data), color='green')
axes[0, 0].set_title('Cumulative Sum')
axes[0, 0].set_xlabel('Index')
axes[0, 0].set_ylabel('Cumulative Sum')
axes[0, 0].grid(True)

axes[0, 1].scatter(range(len(data)), data, color='blue')
axes[0, 1].set_title('Scatter Plot with Noise')
axes[0, 1].set_xlabel('Index')
axes[0, 1].set_ylabel('Value')
axes[0, 1].grid(True)

axes[1, 0].axis('off')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()