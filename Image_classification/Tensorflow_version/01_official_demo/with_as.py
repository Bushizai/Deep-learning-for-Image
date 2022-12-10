

class handsomeb0:
    def __enter__(self):
        print("进入 enter 方法")
        return 'handsomeb'

    def __exit__(self, type,value,trace):
        print("进入 exit 方法")

def get_handsomeb0():
    return handsomeb0()

with get_handsomeb0() as h:
    print("h:", h)    # h调用的是enter的return值
    print("#===========================#")

# ================================================== 【with as的用处】 ================================================= #
class Handsomeb:
    def __enter__(self):
        print('进入 enter 方法')
        return self

    def __exit__(self, type, value, trace):
        print("进入 exit 方法")
        print("type", type)
        print("value", value)
        print("trace", trace)
        return True

    def cal(self):
        return 100/0

def get_handsomeb():
    return Handsomeb()

with get_handsomeb() as h:
    h.cal()