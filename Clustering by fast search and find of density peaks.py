__author__ = 'mgchbot'
import math
import numpy as np
class cluster:

    def ComputeLocalDensity_Parsen(self,data=[],H=0.1):#计算每个点的概率密度(parsen方窗)
        r = H/2
        result=[]
        for line1 in data:
            count=0
            for line2 in data:
                i=0
                flag=0
                for i in range(len(line1)):
                    if math.fabs(line2[i]-line1[i])<r:
                        flag=0
                        continue
                    else:
                        flag=1
                        break
                if flag==0:
                    count = count +1
            p = float(count)/len(data)/H
            result.append(p)
        return result

    def ComputeLocalDensity_Knn(self,data=[],K=5):#计算每个点的概率密度(knn)
        pass

    def ComputeNearestHigherDensity(self,data=[]):#计算离每个点最近的大于这个点的概率密度的点
        result=[]
        for line1 in data:
            mind=100000
            minindex=0
            for i,line2 in enumerate(data):
                te=np.linalg.norm(np.array(line1)-np.array(line2))
                if not te==0:
                    if mind>te:
                        mind=te
                        minindex=i
            result.append([mind,minindex])
        return result

    def FindOutlier(self,density,mind):
        #根据ComputeNearestHigherDensity和 ComputeLocalDensity_Parsen的结果，计算离群点，
        density=np.array(density)
        mind=np.array(mind)
        # print density
        # print mind
        density=(density-min(density))/(max(density)-min(density))
        mind=(mind-min(mind))/(max(mind)-min(mind))
        dt={}
        for i,d in enumerate(density):
            dt[i]=d+mind[i]
        result=sorted(dt.iteritems(), key=lambda d:d[1], reverse = True)
        return result

    def GetClusters(self,data=[],H=0.1):
        # 离群点就是聚类中心，对其他点，找到离自己最近的大于这个点的概率密度的点的类作为自己的类
        density=self.ComputeLocalDensity_Parsen(data,H)
        mind=self.ComputeNearestHigherDensity(data)
        outliers=self.FindOutlier(density,np.matrix(mind)[:,0])
        maxplace=0
        maxd=0
        # print outliers
        for i in range(len(outliers)-1):
            if outliers[i][1][0]-outliers[i+1][1][0]>maxd:
                maxd=outliers[i][1][0]-outliers[i+1][1][0]
                maxplace=i
        centers=[]
        dt={}
        c=1
        for i in range(maxplace):
            if i>0:
                if outliers[i][1][0]==outliers[i-1][1][0]:
                    dt[outliers[i][0]]=dt[outliers[i-1][0]]
                else:
                    c+=1
                    dt[outliers[i][0]]=c
                    centers.append(outliers[i][0])
            else:
                centers.append(outliers[i][0])
                dt[outliers[i][0]]=c

        flag=True
        count=0
        while(flag):
            flag=False
            lastcount=count
            count=0
            for i,line in enumerate(data):
                if not dt.has_key(i):
                    count+=1
                    dt[i]=0
                    flag=True

                else:
                    if dt[i]==0:
                        count+=1
                        flag=True
                        if not dt[mind[i][1]]==0:
                            dt[i]=dt[mind[i][1]]
                            # print dt[i]
            if lastcount==count:
                break

        for key in dt.keys():
            if dt[key]==0:
                mindd=100000
                minindex=0
                for key2 in dt.keys():
                    if dt[key2]>0:
                        te=np.linalg.norm(np.array(data[key])-np.array(data[key2]))
                        if not te==0:
                            if mindd>te:
                                mindd=te
                                minindex=key2
                # print dt[minindex]
                dt[key]=dt[minindex]
        return dt,centers

