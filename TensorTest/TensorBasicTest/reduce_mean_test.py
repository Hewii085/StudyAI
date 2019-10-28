import tensorflow as tf

#reduce_mean은 reduce_sum과 메커니즘은 동일 하나 합산 연산이 아닌 평균을 구한다

def test_two():
    bias = [[4.],[4.]]
    x_holder = tf.placeholder(tf.float32, shape=[2,2])
    logit = tf.reduce_mean(x_holder,axis=0) + bias

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = [[1.,1.],[2.,2.]]


    print(sess.run(logit,feed_dict={x_holder:x}))


def test_one():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    x = [[1.,1.],[2.,2.]]

    print(tf.reduce_mean(x).eval(session=sess))
    print(tf.reduce_mean(x,axis=0).eval(session=sess))
    print(tf.reduce_mean(x,axis=0,keep_dims=True).eval(session=sess))
    print(tf.reduce_mean(x,axis=1).eval(session=sess))


test_two()