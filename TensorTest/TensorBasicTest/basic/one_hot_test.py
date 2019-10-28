import tensorflow as tf

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x = [[0],[1],[2]]
x = tf.one_hot(x,depth=5,on_value=5.0).eval(session=sess)
#indices on_Value가 채워질 index 번호.

print(x)
print("x.shape:",x.shape)    