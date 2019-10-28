import tensorflow as tf

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x = [[1,1,1],[1,1,1]]
y = [[2,2,2],[2,1,2]]

tf.rank(x).eval(session=sess)
sumRslt = tf.reduce_sum(x).eval(session=sess) 
#모든 요소 합산 = 1+1+1+1+1+1
print(sumRslt)

sumRslt=tf.reduce_sum(x, axis=0).eval(session=sess)
#1번째 차원 제거하고 2번째 차원에 합산 1+1, 1+1, 1+1
print(sumRslt)

sumRslt = tf.reduce_sum(x,axis=1).eval(session=sess)
#2번째 차원 제거하고 1번째 차원에 합산 1+1+1, 1+1+1
print(sumRslt)

sumRslt = tf.reduce_sum(x,axis=1,keep_dims=True).eval(session=sess)
#2번째 차원 제거하고 1번째 차원에 합산 1+1+1, 1+1+1
#keep_dims로 차원 유지.
print(sumRslt)

sumRslt = tf.reduce_sum(x,axis=-1).eval(session=sess)
#마지막에서 1번째 차원 제거  1+1+1, 1+1+1
print(sumRslt)

sumRslt = tf.reduce_sum(x,axis=-2).eval(session=sess)
#마지막에서 2번째 차원 제거  1+1, 1+1, 1+1
print(sumRslt)
