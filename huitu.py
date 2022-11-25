import matplotlib.pyplot as plt


def topk():
    x = [0, 40, 50, 60]
    yr1 = [27.4, 31.6, 31.5, 31.8]
    yr2 = [9.8, 11.9, 11.9, 11.7]
    yrl = [23.8, 24.8, 24.9, 25.1]
    ybleu = [8.7, 7.6, 7.8, 7.6]
    ybscore = [88.7, 89.0, 89.0, 89.0]
    ytb = [0.38, 0.39, 0.38, 0.31]
    yc = [84.8, 90.0, 90.6, 90.0]
    
    return x, yr1, yr2, yrl, ybleu, ybscore, ytb, yc  

x, yr1, yr2, yrl, ybleu, ybscore, ytb, yc = topk()

plt.plot(x, yr1, color='orangered', marker='o', linestyle='-', label='R-1')
plt.plot(x, yr2, color='blueviolet', marker='D', linestyle='-.', label='R-2')
plt.plot(x, yrl, color='green', marker='*', linestyle=':', label='R-L')
plt.plot(x, ybleu, color='k', marker='p',  linestyle=':', label='BLEU')
plt.plot(x, ybscore, color='b', marker='X', linestyle='-', label='BScore')
plt.plot(x, ytb, color='m', marker='+', linestyle='-', label='TB')
plt.plot(x, yc, color='orange', marker='^', linestyle='-', label='RS')


plt.legend(bbox_to_anchor=(1.05, 1), loc=2) 
plt.xlabel("Top-k") 
plt.ylabel("Value") 
plt.savefig("Topk.png",bbox_inches='tight')