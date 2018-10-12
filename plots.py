import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

final_1A = pd.read_csv("final_1A.csv")
final_1B = pd.read_csv("final_1B.csv")
final_2A = pd.read_csv("final_2A.csv")
final_2B = pd.read_csv("final_2B.csv")

final_1B.sort_values(by=['modal'],inplace = True)
final_2A.sort_values(by=['modal'],inplace = True)

plt.scatter(x=final_1A['passenger'],y= final_1A['evg'], label = 'group 1 dataset A average')
plt.scatter(x=final_2A['passenger'],y= final_2A['modal'], label = 'dataset A modal prediction')
plt.xticks(final_1A['passenger'])
plt.ylabel('average')
plt.legend(loc='best')
plt.savefig('1A',bbox_inches='tight')

pearsonr(final_1A['evg'],final_2A['modal'])

plt.scatter(x=final_2B['passenger'],y= final_2B['evg'], label = 'group 2 dataset B average')
plt.scatter(x=final_1B['passenger'],y= final_1B['modal'], label = 'dataset B modal prediction')
plt.xticks(final_2B['passenger'])
plt.ylabel('average')
plt.legend(loc='best')
plt.savefig('2B',bbox_inches='tight')

pearsonr(final_2B['evg'],final_1B['modal'])

plt.scatter(x=final_1B['passenger'],y= final_1B['evg'], label = 'group 1 dataset B average')
plt.scatter(x=final_1B['passenger'],y= final_1B['modal'], label = 'dataset B modal prediction')
plt.xticks(final_1B['passenger'])
plt.ylabel('average')
plt.legend(loc='best')
plt.savefig('1B',bbox_inches='tight')

pearsonr(final_1B['evg'],final_1B['modal'])

plt.scatter(x=final_2A['passenger'],y= final_2A['evg'], label = 'group 2 dataset A average')
plt.scatter(x=final_2A['passenger'],y= final_2A['modal'], label = 'dataset A modal prediction')
plt.xticks(final_2A['passenger'])
plt.ylabel('average')
plt.legend(loc='best')
plt.savefig('2A',bbox_inches='tight')

pearsonr(final_2A['evg'],final_2A['modal'])

allBdata = final_1B.rename(columns={"evg": "group1_evg"})
allBdata['group2_evg'] = final_2B['evg']
allBdata.sort_values(by=['modal'],inplace = True)

plt.plot(allBdata['modal'],allBdata['group1_evg'],label = 'group 1 dataset B average')
plt.plot(allBdata['modal'],allBdata['group2_evg'],'r',label = 'group 2 dataset B average')
plt.legend(loc='best')
plt.plot(allBdata['modal'],allBdata['group1_evg'],'bo')
plt.plot(allBdata['modal'],allBdata['group2_evg'], 'ro')
plt.xlabel('modal prediction')
plt.savefig('allB',bbox_inches='tight')

allAdata = final_2A.rename(columns={"evg": "group2_evg"})
allAdata['group1_evg'] = final_1A['evg']
allAdata.sort_values(by=['modal'],inplace = True)

plt.plot(allAdata['modal'],allAdata['group1_evg'],label = 'group 1 dataset A average')
plt.plot(allAdata['modal'],allAdata['group2_evg'],'r',label = 'group 2 dataset A average')
plt.legend(loc='best')
plt.plot(allAdata['modal'],allAdata['group1_evg'],'bo')
plt.plot(allAdata['modal'],allAdata['group2_evg'], 'ro')
plt.xlabel('modal prediction')
plt.savefig('allA',bbox_inches='tight')

plt.plot(allBdata['modal'],allBdata['group1_evg'],'g',label = 'group 1 dataset B average')
plt.plot(allAdata['modal'],allAdata['group2_evg'],'c',label = 'group 2 dataset A average')
plt.legend(loc='best')
plt.plot(allBdata['modal'],allBdata['group1_evg'],'go')
plt.plot(allAdata['modal'],allAdata['group2_evg'], 'co')
plt.xlabel('modal prediction')
plt.savefig('explainedBlackBox',bbox_inches='tight')