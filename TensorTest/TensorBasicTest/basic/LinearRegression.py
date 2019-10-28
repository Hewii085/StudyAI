import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]

#Variable -> tensroflow에서 사용 되는 variable / trainable이라 할 수 있다.
#tf.random_normal([n]) => n : shape. 아래의 경우 1차원의 shape 생성.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


hypothesis = x_train * W + b


cost = tf.reduce_mean(tf.square(hypothesis - y_train))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(2001):
    sess.run(train)

    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
