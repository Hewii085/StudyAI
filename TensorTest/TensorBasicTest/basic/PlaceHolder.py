import tensorflow as tf

#Tensor Flow는 Graph 기반의 연산 구조로
#tensorflow를 이용하여 변수를 생성하거나 하면 기본적으로 정보를 담을 수 있는
#Node가 생긴다. 해당 노드를 이용하여 연산의 흐름이나 로직 자체를 
#연결된 간선을 통하여 연산이 되며 노드 안에는 연산 방법이나 
#기타 정보들이 담겨져 있다.
#즉 Tensorflow를 활용하려면 기본적으로 node를 활용 하여야한다.

a = tf.placeholder(tf.float32) # a라는 비어있는 노드를 생성
b = tf.placeholder(tf.float32) # b라는 비어있는 노드를 생성

adder_node = a+b # a 노드와 b노드를 덧셈을 하는 노드를 생성

sess = tf.Session()
#session run을 통하여 adder_node를 실행 하면 feed_dict를 통하여 값을 전달.
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b:[2,4]}))