import tensorflow as tf

#reduce_prod은 reduce_sum과 메커니즘은 동일 하나 곱셈을 연산을 한다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x = [[3.,3.],
     [2.,2.]]

print(tf.reduce_prod(x).eval(session=sess))
print(tf.reduce_prod(x,axis=0).eval(session=sess))
print(tf.reduce_prod(x,axis=0,keep_dims=True).eval(session=sess))
print(tf.reduce_prod(x,axis=1).eval(session=sess))


