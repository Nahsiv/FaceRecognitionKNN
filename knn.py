import numpy as np
from matplotlib import pyplot as plt

plt.style.use('seaborn-v0_8')
# You can also shift the scale by generating the number using np.random.rand
np.random.seed(12)
l0 = np.random.rand(180, 2)
l0 = 12 + l0 * 3
l1 = np.random.rand(180, 2)
l1 = 15 + l1 * 3
l1 = np.concatenate((l0, l1))
l2 = np.array([0 for _ in range(180)] + [1 for _ in range(180)])
print(l1.shape)
print(l2.shape)
# We have our data generated
plt.scatter(l1[:, 0], l1[:, -1], c=l2)
plt.show()

query_x = np.array([0, 0])
plt.scatter(query_x[0], query_x[1], color='red')
plt.show()


def distance(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))


def knn(l1, l2, query_x, k=5):
    ans = []
    m = l2.shape[0]
    for i in range(m):
        d = distance(query_x, l1[i])
        ans.append((d, l2[i]))
    ans = sorted(ans)
    ans = ans[:k]
    ans = np.array(ans)
    print(ans)
    new_ans = np.unique(ans[:, 1],return_counts=True)
    index = new_ans[1].argmax()
    prediction = new_ans[index][0]
    print(new_ans)
    return prediction


ans1 = knn(l1, l2, query_x)
print(ans1)

