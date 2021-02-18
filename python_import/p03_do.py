# 같은 폴더 내에 있는 .py 파일을 불러올 수 있다.

import p01_car  # p01_car.py에 있는 걸 모두 가져온다.
import p02_tv   # p02_tv.py에 있는 걸 모두 가져온다.

# >> __pycache__ 가 생성된다. 

print("=================================")

# import 되어 있는 파일 내에 있는 함수를 사용할 수 있다.
p01_car.drive()
p02_tv.watch()
