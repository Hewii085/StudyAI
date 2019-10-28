import tensorflow as tf

x_data =[[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32, shape=[None, 2])#??? what is placeholder
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([2,1], name='weight'))
# tf.random_normal([2,1], name='weight') => x의 한행의 data가 2개 , y의 데이터는 1개이다.
# x의 Input으로 y 의 ouput 결과값을 토대로 그다음 데이터에 대한 예측을 위함이기 때문에,
# 어떤 값들의 input으로 어떤 output이 나오고 그다음 데이터를 예측한다.
b = tf.Variable(tf.random_normal([1]), name='bias')
#bias는 output data의 갯수와 같다.

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + ( 1 - Y) * tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracㅛ = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _= sess.run([cost,train],feed_dict={X: x_data, Y:y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})