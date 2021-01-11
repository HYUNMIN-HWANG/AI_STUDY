import numpy as np

# 필요한 패키지와 라이브러리를 가져옴
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
# C:\Users\ai\Anaconda3\Lib\site-packages\matplotlib\mpl-data\fonts\ttf
rc('font', family=font_name)

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

x = np.array([1,2,3,4,5])
y = np.array([6,7,8,9,10])

plt.plot(x,y,marker='.', c='red')
mpl.rcParams['axes.unicode_minus'] = False
plt.title('시간별 가격 추이')
plt.ylabel('주식 가격')
plt.xlabel('시간(분)')

print ('버전: ', mpl.__version__)
print ('설치 위치: ', mpl.__file__)
print ('설정 위치: ', mpl.get_configdir())
print ('캐시 위치: ', mpl.get_cachedir())


# 버전:  3.3.2
# 설치 위치:  C:\Users\ai\Anaconda3\lib\site-packages\matplotlib\__init__.py
# 설정 위치:  C:\Users\ai\.matplotlib
# 캐시 위치:  C:\Users\ai\.matplotlib


font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
print(len(font_list))
print(font_list[:10])
f = [f.name for f in fm.fontManager.ttflist]
print([(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name])
# C:\Users\ai\Anaconda3\Lib\site-packages\matplotlib\mpl-data

