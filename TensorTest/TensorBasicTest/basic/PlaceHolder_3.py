import tensorflow as tf

def train_pip(cost,train,X,Y):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(2001):
            cost_val, _= sess.run([cost,train],feed_dict={X:[1,2,3], Y:[1,2,3]})

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

train_pip(cost,train,X,Y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(hypothesis.eval(feed_dict={X:[5]}))



# import tensorflow as tf

# def train_pip(cost,W,b,train,X,Y):
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())

#     for step in range(2001):
#         cost_val, W_val, b_val, _= sess.run([cost,W,b,train],feed_dict={X:[1,2,3], Y:[1,2,3]})

#     return sess

# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)

# W = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = X * W + b

# cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

# sess=train_pip(cost,W,b,train,X,Y)

# print(sess.run(hypothesis,feed_dict={X:[5]}))