def drive() :
    print("운전하다.")

# 해당 파일 모듈 이름이 '메인'일 때만 함수가 실행되도록 if문을 추가시킨다.
# 이 파일을 댕겨간 파일에서는 모듈 이름이 파일 이름으로 변경되었기 때문에 함수가 실행되지 않는다.
if __name__ == '__main__' :
    drive()