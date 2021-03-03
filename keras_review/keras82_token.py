from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# text = '베링 해가 훅, 사라졌다. 백색 어둠이 그 자리를 훅 채웠다. 해가 훅 바람이 눈발을 휘몰며 불어치고 암벽 같은 빙무가 세상을 가뒀다.'
docs = ['너무 좋아요','하기 정말 싫어요','정말 즐거운 시간이었네요','너무 행복해요','정말 좋은 사람이군요',
        '그만해','내일도 즐거운 하루를 보낼거에요','너무 좋은 시간',
        '이상한 사람인데','역겨우니까 저리 치워','보기 싫으니까 저리 가','정말 재미없어']

# 긍정 1, 부정 0
labels = np.array([1,0,1,1,1,0,1,1,0,0,0,0])  # y

token = Tokenizer()
token.fit_on_texts(docs)

print(token.word_index)
# {'정말': 1, '너무': 2, '즐거운': 3, '좋은': 4, '저리': 5, '좋아요': 6, '하기': 7, '싫어요': 8, '시간이었네요': 9, '행복해요': 10, '사람이군요': 11, 
# '그만해': 12, '내일도': 13, '하루를': 14, '보낼거에요': 15, '시간': 16, '이상한': 17, '사람인데': 18, '역겨우니까': 19, '치워': 20, '보기': 21, '싫으니까': 22, '가': 23, '재미없어': 24}


x = token.texts_to_sequences(docs)
print(x)
# [[2, 6], [7, 1, 8], [1, 3, 9], [2, 10], [1, 4, 11], [12], [13, 3, 14, 15], [2, 4, 16], [17, 18], [19, 5, 20], [21, 22, 5, 23], [1, 24]]

# preprocessing
# x
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=4)

print(pad_x)
print(pad_x.shape)  # (12, 4)

print(len(np.unique(pad_x)))    # 25

# Modeling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Conv1D

model = Sequential()
model.add(Embedding(input_dim=30, output_dim=16, input_length=4))
model.add(Conv1D(32, 3))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=50)

results = model.evaluate(pad_x, labels)
print("loss : ", results[0])
print("acc : ", results[1])

# loss :  0.26044127345085144
# acc :  1.0