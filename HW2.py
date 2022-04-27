{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "54a5607c-f683-4916-b1c4-cef98e173392",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM,Dense,Embedding\n",
    "from tensorflow.keras.losses import sparse_categorical_crossentropy\n",
    "from keras.utils import np_utils\n",
    "from keras.layers import Dropout\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "722e8de0-8eda-42ff-a111-ae56c50946a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "names_draft = []\n",
    "with open('names.txt') as file:\n",
    "    for line in file:\n",
    "        names_draft.append(line[0:line.find(',')])\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "320dbc43-c421-4a35-857a-037fbfe1f6eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "names = []\n",
    "for i in names_draft:\n",
    "        names.append(i.lower())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "afbe0be5-abab-4f5a-90de-800970fa3313",
   "metadata": {},
   "outputs": [],
   "source": [
    "# def get_max_str(new_names):\n",
    "#     max_len = max(new_names, key=len)\n",
    "#     return(max_len)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ff834c5a-2330-45ed-9928-5dde70d2aecb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# names_list = [list(name) for name in names]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b380c750-36fe-4093-a11a-5f695fa7d138",
   "metadata": {},
   "outputs": [],
   "source": [
    "# tf.keras.preprocessing.sequence.pad_sequences(names_list, dtype=object, value=' ', padding='post')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7b0e1f4e-9629-4749-a669-8d37e6804512",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "emma>           \n"
     ]
    }
   ],
   "source": [
    "max_len = max(names, key = len)\n",
    "input_names = []\n",
    "output_names = []\n",
    "# input_names_padded = []\n",
    "# output_names_padded = []\n",
    "for i in names:\n",
    "        # input_names.append(\"<\" + i)\n",
    "        # output_names.append((i + \">\") )\n",
    "        input_names.append(\"<\" + i.ljust(len(max_len)))\n",
    "        output_names.append((i + \">\").ljust(len(max_len ) + 1))        \n",
    "\n",
    "        \n",
    "print(output_names[0])\n",
    "\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "d86660d0-562d-4339-b901-ed9856e67e52",
   "metadata": {},
   "outputs": [],
   "source": [
    "one_hot = []\n",
    "a = []\n",
    "for i in input_names:\n",
    "    a = np.zeros((16,28))\n",
    "    for char_ind, j in enumerate(i):\n",
    "        posit = ord(j) - ord(\"a\")       \n",
    "        if(j == \"<\"):\n",
    "            posit = 26\n",
    "        elif(j == \">\"):\n",
    "            posit = 27\n",
    "        if(j != ' '):    \n",
    "            a[char_ind, posit] = 1\n",
    "    one_hot.append(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ae934c03-740c-4f13-95dc-78a2a09496b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "one_hot_output = []\n",
    "a = []\n",
    "for i in output_names:\n",
    "    a = np.zeros((16,28))\n",
    "    for char_ind, j in enumerate(i):\n",
    "        posit = ord(j) - ord(\"a\")       \n",
    "        if(j == \"<\"):\n",
    "            posit = 26\n",
    "        elif(j == \">\"):\n",
    "            posit = 27\n",
    "        if(j != ' '):    \n",
    "            a[char_ind, posit] = 1\n",
    "    one_hot_output.append(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0bc46c49-a0e3-40fb-8b5d-f53b79b975a4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 1.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0.]]\n"
     ]
    }
   ],
   "source": [
    "print(one_hot_output[0])\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "737a8911-3c54-4129-8750-77cfd36607f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "input_shape = a.shape\n",
    "model = Sequential()\n",
    "model.add(tf.keras.layers.Masking(mask_value=np.zeros((28,)),\n",
    "                                  input_shape=(len(max_len)+1, 28)))\n",
    "model.add(LSTM(128, return_sequences=True,\n",
    "               input_shape=input_shape))\n",
    "model.add(LSTM(128, return_sequences=True))\n",
    "model.add(Dense(28, activation='softmax'))\n",
    "\n",
    "model.compile(optimizer = \"adam\", loss= \"CategoricalCrossentropy\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "72357b3d-8564-45a8-a9e5-f2ad239d34f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/15\n",
      "801/801 [==============================] - 28s 31ms/step - loss: 1.0718 - val_loss: 1.0852\n",
      "Epoch 2/15\n",
      "801/801 [==============================] - 23s 29ms/step - loss: 0.9601 - val_loss: 1.0355\n",
      "Epoch 3/15\n",
      "801/801 [==============================] - 23s 29ms/step - loss: 0.9286 - val_loss: 1.0180\n",
      "Epoch 4/15\n",
      "801/801 [==============================] - 24s 30ms/step - loss: 0.9061 - val_loss: 1.0083\n",
      "Epoch 5/15\n",
      "801/801 [==============================] - 24s 29ms/step - loss: 0.8876 - val_loss: 0.9935\n",
      "Epoch 6/15\n",
      "801/801 [==============================] - 24s 29ms/step - loss: 0.8715 - val_loss: 0.9959\n",
      "Epoch 7/15\n",
      "801/801 [==============================] - 25s 31ms/step - loss: 0.8579 - val_loss: 0.9849\n",
      "Epoch 8/15\n",
      "801/801 [==============================] - 24s 30ms/step - loss: 0.8461 - val_loss: 0.9828\n",
      "Epoch 9/15\n",
      "801/801 [==============================] - 24s 30ms/step - loss: 0.8355 - val_loss: 0.9818\n",
      "Epoch 10/15\n",
      "801/801 [==============================] - 24s 30ms/step - loss: 0.8254 - val_loss: 0.9785\n",
      "Epoch 11/15\n",
      "801/801 [==============================] - 24s 30ms/step - loss: 0.8172 - val_loss: 0.9799\n",
      "Epoch 12/15\n",
      "801/801 [==============================] - 24s 30ms/step - loss: 0.8091 - val_loss: 0.9801\n",
      "Epoch 13/15\n",
      "801/801 [==============================] - 24s 30ms/step - loss: 0.8014 - val_loss: 0.9832\n",
      "Epoch 14/15\n",
      "801/801 [==============================] - 24s 30ms/step - loss: 0.7946 - val_loss: 0.9804\n",
      "Epoch 15/15\n",
      "801/801 [==============================] - 24s 30ms/step - loss: 0.7883 - val_loss: 0.9829\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x16934f5b0>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(np.array(one_hot), np.array(one_hot_output), validation_split=0.2, epochs = 15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "38019b23-df5d-48ff-be49-c843feaeda95",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"trained_model.h5\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "9ff857bd-9e3e-4a0a-9526-7e539b8465c1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1, 16, 28)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict(x).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "112a9323-61ea-4492-aeff-cde9843cd94b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_trained_model(path):\n",
    "    model = load_model(path) \n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "6f995015-9102-4143-92ea-0d8bc5a0c309",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate(N):\n",
    "    model = get_trained_model(path)\n",
    "    \n",
    "    name = []\n",
    "    start_char = np.zeros((28,))\n",
    "    start_char[26] = 1\n",
    "    \n",
    "    x = np.zeros((1, len(max_len)+1, 28))\n",
    "    i = 0\n",
    "    \n",
    "    while i < N:\n",
    "        probs = list(model.predict(x)[0,i])\n",
    "        probs = probs / np.sum(probs)\n",
    "        index = np.random.choice(range(28), p=probs)\n",
    "        if index == 27:\n",
    "            break\n",
    "        character = list('abcdefghijklmnopqrstuvwxyz<>')[index]\n",
    "        name.append(character)\n",
    "        x[0, i+1, index] = 1\n",
    "        i += 1\n",
    "    return name"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "171cccfb-9104-4e8d-a439-0e6a2154e8cf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['l', 'a', 'y', 'k', 'e']"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "make_name(14)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "c7d17024-1979-4c86-a4c2-945a7b5a6d81",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "name"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "dc1e31e2-377f-4468-a755-ec50024289e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " lstm (LSTM)                 (None, 16, 128)           80384     \n",
      "                                                                 \n",
      " lstm_1 (LSTM)               (None, 16, 128)           131584    \n",
      "                                                                 \n",
      " dense (Dense)               (None, 16, 28)            3612      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 215,580\n",
      "Trainable params: 215,580\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b358803d-0196-42cc-8ded-76a8e169e75c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " masking (Masking)           (None, 16, 28)            0         \n",
      "                                                                 \n",
      " lstm (LSTM)                 (None, 16, 128)           80384     \n",
      "                                                                 \n",
      " lstm_1 (LSTM)               (None, 16, 128)           131584    \n",
      "                                                                 \n",
      " dense (Dense)               (None, 16, 28)            3612      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 215,580\n",
      "Trainable params: 215,580\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "from RNNModel import create_model, generate_name, model_link\n",
    "um = create_model(16)\n",
    "um.summary() # print out summary of the model um.fit(right_shifted_train_data, training_data, epochs=5, batch_size=128) # note train data here are 3d arrays with shape [batch, maxlen, 28].\n",
    "  \n",
    "# Download saved model using url in variable “model_link”\n",
    "# Print 5 generated names (We expect there are different names among the 5.) for i in range(5):\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "9746d946-5567-4c3e-92ed-6fca7bf16555",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "801/801 [==============================] - 29s 32ms/step - loss: 1.0751 - val_loss: 1.0803\n",
      "Epoch 2/2\n",
      "801/801 [==============================] - 23s 29ms/step - loss: 0.9617 - val_loss: 1.0433\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x175ec1040>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "um.fit(np.array(one_hot), np.array(one_hot_output), validation_split=0.2, epochs = 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "18bb5db2-437b-423a-90e0-7dc7a3643f86",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "rajhany\n",
      "daizee\n",
      "baobai\n",
      "nayoa\n",
      "agvenemah\n"
     ]
    }
   ],
   "source": [
    "for i in range(5):\n",
    "    print(generate_name(16))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92acf5d5-5a47-46f8-adaa-d2b26d99b799",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
