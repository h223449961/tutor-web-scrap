# -*- coding: utf-8 -*-
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
import requests
from styleframe import StyleFrame, Styler
from bs4 import BeautifulSoup
import pandas as pd
import tkinter as tk
import re
import numpy as np
import math
pd.set_option('mode.chained_assignment', None)
class NeuralNetwork():
    def __init__(self):
        # 随机数生成的种子随机数生成的种子
        np.random.seed(1)
        # 将权重转换为值为-1到1且平均值为0的3乘1矩阵
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # 定义signoid函数的导数
    def sigmoid(self, x):
        x = x.astype(float)
        return (1 / (1 + np.exp(-x)))
    # 计算sigmoid函数的导数
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # 训练
    def train(self, train_inputs, train_outputs,i): # 输入 输出 迭代次数
        # 训练模型在不断调整权重的同时做出准确预测
        for iteration in range(i):
            # 通过神经元提取训练数据
            output = self.think(train_inputs)
            # 反向传播错误率
            error = train_outputs - output
            # 进行权重调整
            adjustments = np.dot(train_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights = self.synaptic_weights + adjustments
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
root = tk.Tk()
root.geometry("1600x800")
radioValue = tk.IntVar()
exfile_frame = tk.Frame(root)
exfile_frame.pack(side=tk.TOP)
exfile_label = tk.Label(exfile_frame, text='快速點擊爬取',font=(24),fg = "#30aabc")
exfile_label.pack(side=tk.LEFT)
# 臺東
tung = '102200'
# 宜蘭
il = '100400'
# 花蓮
hua = '102100'
# 桃園
yuan = '100500'
# 臺北
taip = '100100'
# 新北
newp = '100200'
# 基隆
ki = '100300'
# 新竹市
shn = '100600'
# 新竹縣
shnn = '100700'
# 苗栗
mia = '100800'
# 彰化
cha = '101200'
# 南投
to  = '101100'
# 臺中
taic = '100900'
# 雲林
yun = '101300'
# 嘉義市
chia = '101400'
# 嘉義縣
chiaa = '101500'
# 連江
lian = '102500'
# 金門
mon = '102400'
# 澎湖
pon = '102300'
# 臺南
nan = '101600'
# 高雄
kao = '101800'
# 屏東
pin = '102000'
def validate():
    value = radioValue.get()
    global mode
    if (value == 1):
        mode = tung
    if( value == 2):
        mode = lian
    if( value == 3):
        mode = hua
    if (value == 4):
        mode = mia
    if( value == 5):
        mode = pon
    if( value == 6):
        mode = pin
    if (value == 7):
        mode = kao
    if( value == 8):
        mode = shn
    if( value == 9):
        mode = shnn
def oas():
    global mode
    df01 = pd.DataFrame(columns=['名字','科目','科目加權分','學習狀態','學校','科系'])
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:67.0) Gecko/20100101 Firefox/67.0',}
    
    #mode = yuan
    html = requests.get(url='https://tutor.1111.com.tw/teacher/teacherSearch.asp?newSearch=search&Item1=1&sCity0='+mode,headers=headers).content
    soup = BeautifulSoup(html,'lxml')
    nams = soup.find_all('div',class_='colContent')
    for nam in nams:
        lst = [_.get_text(strip=True) for _ in nam.find('ul', {'class': 'leftFloat'}).find_all('li')]
        nam = lst[0]
        print(nam[5:22])
        sub = lst[2]
        sub_list = sub[5:37]
        result = len(sub_list.split(','))
        print(sub_list,result)
        bch = lst[6]
        bacho = bch[5:99]
        a, b,c = bacho.split('‧')
        print(a+' and '+b+' and '+c+'\n')
        s01 = pd.Series([nam[5:22],sub_list,result,c,a,b], index=['名字','科目','科目加權分','學習狀態','學校','科系'])
        df01 = df01.append(s01, ignore_index=True)
        df01.index = df01.index+1
    statumap =  {'博士(畢業)':6,'博士(肄業)':5.3,'博士(就學中)':5,'碩士(畢業)':4,'碩士(就學中)':3,
                  '大學(畢業)':2,'大學(就學中)':1,'高中職(就學中)':0.7,'專科(畢業)':0.6}
    scholmap =  {
                 '加拿大曼尼托巴大學University of Manitoba (加拿大)':104,'Savannah College of Art and Design':89.2,
                 'The University of New South Wales, Sydne':103.7,'倫敦大學國王學院':106.7,
                 'The University of Manchester':106.8,'LaSalle College Vancouver':86.9,
                 'University of Illinois Urbana-Champaign':106,
                 '天津大學':99,'英屬哥倫比亞大學(夜)':109,'昆士蘭大學':103.9,'日本國立新潟大學':102,'楊百翰':101,
                 '台灣大學':100,'國立臺灣大學':100,'國立台灣大學':100,'國立交通大學':98,'國立成功大學':97,
                 '國立清華大學':99,'清華大學':99,
                 '政治大學':96,'國立政治大學':96,
                 '國立台北大學':89.37,'台北市立大學':79,'市立台北師範大學數學系':79,'國立台灣藝術大學':89,'國立台北藝術大學':89.1,
                 '臺北醫學大學':86.2,'高雄醫學大學':83,'中國醫藥大學(北港校區)':82,'中山醫學大學':82.2,
                 '國立臺灣海洋大學':81.2,'國立台灣海洋大學':81.2,
                 '國立東華大學':71.64,'東華大學':71.64,
                 '國立宜蘭大學':71.68,'宜蘭大學':71.68,
                 '國立暨南國際大學':68,'宜蘭大學':71.68,
                 '國立嘉義大學':67.6,'嘉義大學':67.6,
                 '國立臺南大學':67,'國立臺南藝術大學':76,
                 '國立屏東大學':60,'國立屏東科技大學':58,
                 '國立金門大學':67.4,'金門大學':67.4,
                 '國立臺東大學':65,'台東大學':65,
                 '國立台灣師範大學':92,'國立臺灣師範大學':92,'國立彰化師範大學':86.2,
                 '國立高雄師範大學':86.6,'國立高雄師範大學(假)':86.6,'國立高雄師範大學(燕巢校區)':86.6,
                 '國立中興大學':90,'中山大學':89.9,'中正大學':89.7,'國立中正大學':89.7,'國立中央大學':89.3,
                 '輔仁大學':85.6,'慈濟大學':56.3,'中國文化大學':66,'銘傳大學':79.2,
                 '長榮大學':58.2,'長榮大學(夜)':58.2,
                 '東吳大學':86.8,'東海大學':71.2,'實踐大學':86,'靜宜大學':65,'中原大學':90,'華梵大學':56.8,
                 '崑山科技大學':56.2,'美和科技大學':48,'大仁科技大學':52,'台中商專、朝陽科技大學':62,'建國科技大學':56,'大仁科技大學':52,
                 '國立勤益科技大學':72,'國立臺中科技大學':71.6,
                 '國立臺灣科技大學':92.2,'國立台灣科技大學':92.2,
                 '國立高雄第一科技大學':86,
                 '國立臺北科技大學':92.3,'國立台北工業專科學校(假)':92.3,
                 '國立虎尾科技大學':71.7,'國立雲林科技大學':89,
                 '國立高雄餐旅大學':90,'高雄女中':95,'國立空中大學':60.8,'中壢高中':65
                  }
    df01['學習狀態加權分'] = df01['學習狀態'].map(statumap)
    df01['學校加權分'] = df01['學校'].map(scholmap)
    df01['學校加權分'].fillna(70.9, inplace=True)
    df01['學習狀態加權分'].fillna(2, inplace=True)
    df01['科目加權分'].fillna(2, inplace=True)
    vgrab = []
    tsumrab = []
    average = 0
    tolsum = 0
    for ind in df01.index:
        average = df01['學校加權分'][ind]/109
        tolsum = df01['學校加權分'][ind]*df01['學習狀態加權分'][ind]+df01['科目加權分'][ind]*((df01['學校加權分'][ind])/10)
        vgrab.append(average)
        tsumrab.append(tolsum)
    df01['學校加權分正規化'] = vgrab
    df01['總分'] = tsumrab
    weightsum = 0
    ave = 0
    for ind in df01.index:
        weightsum = weightsum+df01['總分'][ind]
    ave = weightsum/len(df01.index) 
    print(ave)
    
    result = []
    for ind in df01.index:
        if(df01['總分'][ind]>ave):
            result.append("推薦")
        else:
            result.append("普通")
    df01["公式計算"] = result
    #print(df01)
    dft = df01[['科目加權分','學習狀態加權分','學校加權分正規化']]
    #dft = dft.copy
    dftt = df01[['科目加權分','學習狀態加權分','學校加權分正規化']]
    rab = []
    for ind in df01.index:
        if(df01['公式計算'][ind]=='推薦'):
            rab.append(1)
        else:
            rab.append(0)
    dft['rabel'] = rab
    num = dftt.to_numpy()
    numt = num.reshape(1,-1).T
    ranu = dft['rabel'].to_numpy()
    ranut = ranu.reshape(1,-1).T
    def softmax(x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x
    def clo(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    print(num)
    print(ranut)
    b = NeuralNetwork()
    b.train(num, ranut,90000)
    classif = softmax(b.think(num))
    print(classif)
    cround = np.round(classif, 2)
    cro = pd.DataFrame(cround,columns = ['機率'])
    print(cro)
    print('總機率： '+str(np.sum(cround)))
    print('最接近總機率的值： '+str(clo(cround,1)))
    flg = clo(cro,1)
    nlearn = []
    for ind in cro.index:
        if(cro['機率'][ind]==flg):
            nlearn.append('推薦')
        else:
            nlearn.append('普通')
    df01["神經網路"] = nlearn
    print(df01)
    acoun = 0
    bcoun = 0
    altrue = 0
    preci = 0
    for i in df01.index:
        truflg = []
        if(df01['公式計算'][i]==df01['神經網路'][i]):
            acoun = acoun+1
        else:
            bcoun = bcoun+1
        if(df01['神經網路'][i]=='推薦'):
            altrue = altrue +1
        if(df01['公式計算'][i]=='推薦'):
            truflg = df01['公式計算'][i]
        if(truflg==df01['神經網路'][i]):
            preci= preci+1
    print((acoun/(acoun+bcoun))*100,(preci/altrue)*100)
    acpre_label.configure(text ='準確： '+str((acoun/(acoun+bcoun))*100)+' %')
    sf = StyleFrame(df01)
    sf.set_column_width_dict(col_width_dict={("名字"): 22,("科目"): 46,("學習狀態"): 22,("學校"): 27,("科系"): 48,("學校加權分正規化"): 29,("總分"): 26})
    sname = 'tog.xlsx'
    output = sf.to_excel(sname).save()
    df = pd.read_excel(sname)
    df.fillna('官方沒有收集到', inplace=True)
    cols = list(df.columns)
    def treeview_sort_column(tv, col, reverse):
        try:
            l = [float((tv.set(k, col)), k) for k in tv.get_children('')]
        except:
            l = [(tv.set(k, col), k) for k in tv.get_children('')]
        l.sort(reverse=reverse)

    # rearrange items in sorted positions
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)

    # reverse sort next time
        tv.heading(col, command=lambda:
            treeview_sort_column(tv, col, not reverse))
    tree = ttk.Treeview(root)
    tree.pack()
    tree["columns"] = cols
    tree.column("#0", width=0, stretch=False)
    for i in cols:
        # 店名
        tree.column('# 1',width =118,anchor="center")
        # 電話
        tree.column('# 2',width =82,anchor="center")
        # 初始分
        tree.column('# 3',width =0,anchor="center")
        # 總分
        tree.column('# 4',width =0,anchor="center")
        # 素食種類
        tree.column('# 5',width =82,anchor="center")
        # 可配合素食加權分
        tree.column('# 6',width =49,anchor="center")
        # 休息時間
        tree.column('# 7',width =0,anchor="center")
        # 全勤加權分
        tree.column('# 8',width =0,anchor="center")
        # 可手機聯絡加權分
        tree.column('# 9',width =39,anchor="center")
        # 地址
        tree.column('# 10',width =0,anchor="center")
        # 推薦
        tree.column('# 11',width =0,anchor="center")
        # 推薦
        tree.column('# 12',width =0,anchor="center")
        tree.heading(i,text=i,anchor='center')
        tree.heading(i, text=i,command=lambda c=i: treeview_sort_column(tree, c, False))
    for index, row in df.iterrows():
        tree.insert("",'end',text = index,values=list(row))
    tree.place(relx=0,rely=0.4,relheight=0.5,relwidth=1)
r1 = tk.Radiobutton(root,text = "臺東",font=(24),variable=radioValue, value=1,command = validate).pack()
r2 = tk.Radiobutton(root,text = "連江",font=(24),variable=radioValue, value=2,command = validate).pack()
r3 = tk.Radiobutton(root,text = "花蓮",font=(24),variable=radioValue, value=3,command = validate).pack()
r4 = tk.Radiobutton(root,text = "苗栗",font=(24),variable=radioValue, value=4,command = validate).pack()
r5 = tk.Radiobutton(root,text = "澎湖",font=(24),variable=radioValue, value=5,command = validate).pack()
r6 = tk.Radiobutton(root,text = "屏東",font=(24),variable=radioValue, value=6,command = validate).pack()
r7 = tk.Radiobutton(root,text = "高雄",font=(24),variable=radioValue, value=7,command = validate).pack()
r8 = tk.Radiobutton(root,text = "新竹縣",font=(24),variable=radioValue, value=8,command = validate).pack()
r9 = tk.Radiobutton(root,text = "新竹市",font=(24),variable=radioValue, value=9,command = validate).pack()
def com(*args): # 處理事件， *args 表示可變引數  
    global mode
    mode = comboxlist.get()
    if(mode == '桃園'):
        mode = '100500'
    if(mode == '雲林'):
        mode = '101300'
    if(mode == '臺南'):
        mode = '101600'
    if(mode == '基隆'):
        mode = '100300'
    if(mode == '臺北'):
        mode = '100100'
    if(mode == '新北'):
        mode = '100200'
    if(mode == '臺中'):
        mode = '100900'
    if(mode == '嘉義縣'):
        mode = '101500'
    if(mode == '嘉義市'):
        mode = '101400'
    if(mode == '金門'):
        mode = '102400'
    if(mode == '宜蘭'):
        mode = '100400'
    if(mode == '南投'):
        mode = '101100'
    if(mode == '彰化'):
        mode = '101200'
comvalue=tk.StringVar()
comboxlist=ttk.Combobox(root,textvariable=comvalue)
comboxlist["values"]=("桃園","雲林","臺南",'基隆','臺北','新北','臺中','嘉義縣','嘉義市','金門','宜蘭','南投','彰化')  
comboxlist.current(0)
comboxlist.bind("<<ComboboxSelected>>",com) # 繫結事件，（下拉列表框被選中時，繫結綁定的函式）  
comboxlist.pack()
acpre_frame = tk.Frame(root)
acpre_frame.pack(side=tk.TOP)
b1 = tk.Button(acpre_frame, text="爬取（若無特別設定將爬取上個暫存的縣市記錄）",font=(24),command = oas).pack(side=tk.LEFT)
acpre_label = tk.Label(acpre_frame, text='精準度',font=(24),fg = "#C13E43")
acpre_label.pack(side=tk.RIGHT)
root.mainloop()