import tensorflow as tf

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x = [[1,1,1],[1,3,1]]
y = [[2,2,2],[2,1,2]]

tf.rank(x).eval(session=sess)
maxRslt = tf.maximum(x,y).eval(session=sess) 
#각 요소에서 큰 숫자들만 뽑아 새로운 배열을 만들어 리턴한다.
print(maxRslt)
