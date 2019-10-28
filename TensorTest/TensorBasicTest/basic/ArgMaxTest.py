import tensorflow as tf

a = tf.constant([[3, 10, 1, 22]])

session = tf.Session()
print('a:\n', session.run(a))
print('Index Count = ', session.run(tf.rank(a)) )
print('tf.argmax(a, 0): index ', session.run(tf.argmax(a, 1)), 'is Bigger')