# similarity product
sim.pd <- function(d1,d2,ms.pos,sm="dot"){
	d1 = d1[,ms.pos]
	d2 = d2[,ms.pos]
	if(sm=="dot"){
		dd1 = as.matrix(d1)/sqrt(apply((as.matrix(d1))^2,1,sum))
		dd2 = as.matrix(d2)/sqrt(apply((as.matrix(d2))^2,1,sum))
		sim = dd1 %*% t(dd2)
	}else if(sm=="pearson"){
		sim = cor(t(d1),t(d2))
	}
}

# distance from R
dist.r <- function(d1,d2,dm="canberra"){
	# dm = "euclidean", "maximum", "manhattan", "canberra"
	
	d1 = d1
	d2 = d2
	len1 = dim(d1)[1]
	len2 = dim(d2)[1]

	da = rbind(d1,d2)
	da = as.matrix(da)
	tdist = as.matrix(dist(da,method=dm,diag=TRUE,upper=TRUE))[(1:len1),((len1+1):(len1+len2))]
	tdist
}

# score calculation
cscore <- function(d1,d2,align){
	d1.name = as.character(d1$name)
	d2.name = as.character(d2$name)
	total.pos = length(na.omit(match(d1.name,d2.name)))
	total.neg = length(d1.name)*length(d2.name) - total.pos
	
	d1 = d1[align$tid,] # tar
	d2 = d2[align$rid,] # ref
	d1$name = as.character(d1$name)
	d2$name = as.character(d2$name)
	total.match = dim(d1)[1]
	t.p = 0
	for(i in 1:total.match){
		if(d1$name[i] == d2$name[i]){
			t.p = t.p + 1
		}	
	}
	f.p = total.match - t.p
	f.n = total.pos - t.p
	t.n = total.neg - f.p
	t.p.r = t.p/total.pos
	p.p.v = t.p/(t.p+f.p)

	if((t.p.r+p.p.v)==0){
		f1 = 0
	}else{
		f1 = 2*t.p.r*p.p.v/(t.p.r+p.p.v)
	}
		
	rlt=c(t.p.r,p.p.v,f1,t.p,f.p,t.n,f.n)
	rlt
}

sw4pa <- function(para=0.95
		,tar
		,ref
		,method=2
		,out=1
		,sim="pearson"
		,lm="sum"
		,sp.pos=NA
		,vn = c("t1","t2")
	){
	# lm: method of likelihood

	cv.sim = para
	seq1 = tar
	seq2 = ref
		
	# sp.pos
	if(is.na(sp.pos[1])){
		sp.pos = sort(grep("X",colnames(seq1)))
		sp.pos = sp.pos[1]:sp.pos[length(sp.pos)]
	}

	# scoring scheme
	MATCH    =  2 # for letters that match
	MISMATCH = -1 # for letters that mismatch
	GAP      = -1 # for any gap

	# number of samples
	len1 = dim(seq1)[1]
	len2 = dim(seq2)[1]

	# initialization
	score.m = matrix(0,(len2+1),(len1+1))
	pointer.m = matrix("none",(len2+1),(len1+1))
	
	# fill
	max.i     = 0;
	max.j     = 0;
	max.score = 0;

	# similarity and distance
	mcor = sim.pd(seq2[,],seq1[,],ms.pos=sp.pos,sm=sim)
	mdist = dist.r(seq2[,vn],seq1[,vn],dm="euclidean")
	mdist = 1/(1+mdist)

	for(i in 1:len2){
	    	for(j in 1:len1){
	        	diagonal.score = 0 
			left.score = 0
			up.score = 0
        
	        	# calculate match score
			pcor = mcor[i,j]
		      	if(pcor>=cv.sim){
        		    	diagonal.score = score.m[i,j] + MATCH 
	        	}else{
		            diagonal.score = score.m[i,j] + MISMATCH 
        		}
        
		      	# calculate gap scores
        		up.score   = score.m[i,(j+1)] + GAP 
	        	left.score = score.m[(i+1),j] + GAP 
        
	        	if(diagonal.score <= 0 && up.score <= 0 && left.score <= 0){
		      	score.m[(i+1),(j+1)] = 0 
        		    	pointer.m[(i+1),(j+1)] = "none"
	            	next 
	        	}
        
		      	# choose best score
        		if (diagonal.score >= up.score){
	            	if(diagonal.score >= left.score){
        	        		score.m[(i+1),(j+1)] = diagonal.score 
	                		pointer.m[(i+1),(j+1)] = "diagonal" 
		            }else{
                			score.m[(i+1),(j+1)] = left.score 
	                		pointer.m[(i+1),(j+1)] = "left" 
	        	    	}
		        }else{
        		    	if(up.score >= left.score){
                			score.m[(i+1),(j+1)] = up.score 
	                		pointer.m[(i+1),(j+1)] = "up" 
		       	}else{
        		        	score.m[(i+1),(j+1)] = left.score 
                			pointer.m[(i+1),(j+1)] = "left" 
		            }
        		}
        
	        	# set maximum score
        		if(score.m[(i+1),(j+1)] > max.score){
		        	max.i     = i+1 
        		    	max.j     = j+1 
	        	    	max.score = score.m[(i+1),(j+1)]
        		}
	    	}
	}

	# trace-back
	if(max.score==0){
		ll = 10e10
		s.align = NA		
	}else{
		align = c()

		if(method==1){
			j = max.j
			i = max.i
		}else if(method==2){
			j = len1+1 		
			i = len2+1 
			max.score=score.m[(i),(j)]
		}else if(method==3){
			j = len1+1 
			repeat{
				max.score = max(score.m[1:max.i,j])
				if(max.score>0 | j==1){
					break
				}else{
					j = j-1
				}
			}
			i = rev(which(score.m[1:max.i,j]==max.score))[1] 
		}else{
			j = len1+1
			repeat{
				i = rev(which(score.m[1:max.i,j]>0))[1]
				if(j==1){
					break
				}else if(!is.na(i)){
					max.score = score.m[i,j]
					break
				}else{
					j = j-1
				}
			}
		}

		while (1) {
			if(pointer.m[(i),(j)] == "none"){
				if(i<=2){
					break
				}else{
					# find another local maximum
					max.i = i
					max.j = j
					max.score = 0
					if(method==1){
						for(s.i in 1:(i-1)){
						    	for(s.j in 1:(j-1)){
				      				if(score.m[(s.i+1),(s.j+1)] > max.score){
	      		      						max.i = s.i+1
	     		    						max.j = s.j+1
        		    						max.score = score.m[(s.i+1),(s.j+1)]
								}
							}
						}
					}else if(method==2){
						max.i = i
						max.j = j
						max.score = score.m[max.i,max.j]
					}else if(method==3){
						max.j = j
						repeat{
							max.score = max(score.m[1:i,max.j])
							if(max.score>0 | max.j==1){
								break
							}else{
								max.j = max.j-1
							}
						}
						max.i = rev(which(score.m[1:i,max.j]==max.score))[1]
					}else{
						max.j = j
						repeat{
							max.i = rev(which(score.m[1:i,max.j]>0))[1]
							if(max.j==1){
								break
							}else	if(!is.na(max.i)){
								max.score = score.m[max.i,max.j]
								break
							}else{
								max.j = max.j-1
							}
						}
					}

					if(max.j == 1 | max.i == 1){
						break
					}else if(max.score == 0){
						j = j-1
					}else{
 						i = max.i
						j = max.j
					}
				}
			}

			if(pointer.m[i,j] == "diagonal"){
				align = rbind(
						c(
						seq1$t1[j-1],seq1$t2[j-1]
						,seq2$t1[i-1],seq2$t2[i-1]
						,mcor[(i-1),(j-1)]
						,mdist[(i-1),(j-1)]
						,seq1$idx[j-1]
						,seq2$idx[i-1]
						)
						,align
					)
  				i = i-1
				j = j-1
		    	}else if(pointer.m[i,j] == "left"){
        			j = j-1 
	    		}else if(pointer.m[i,j] == "up"){
     	  			i = i-1 
	    		}
		}
	
		align = as.data.frame(align)
		
		# correlation based outlier
		s.align = align[align[,5]>=cv.sim,]

		if(lm=="sum"){
			ll = sum(s.align[,5]) #+ sum(s.align[,6])
			#cat(cv.sim,ll,"\n") 
			ll = -ll
		}else{
			ll = prod(s.align[,5])
		}
	}
	
	if(out==1){
		ll
	}else{		
		s.align
	}
}

# optimal based on multiple shooting
# find the initial
# find.ini <- function(
mult.shoot <- function(
			fun=sw4pa
			,tar,ref
			,method
			,out=1
			,sim
			,lm
			,sp.pos
			,vn
	){
	#cvs = c(.95,rev(seq(.5,.9,by=.1)))
	cvs = c(.5,.55,.6,.65,.7,.75,.8,.85,.9,.93,.95,.97,.99)
	min.id = 1
	min.method = 1
	min.ll = 10e10
	for(i in 1:length(cvs)){
		for(j in 1:3){
			ll = sw4pa(
				para=cvs[i]
				,tar=tar
				,ref=ref
				,method=j
				,out=out
				,sim=sim
				,lm=lm
				,sp.pos=sp.pos
				,vn = vn
			)		
			if(ll<min.ll){
				min.ll = ll
				min.id = i
				min.method = j
			}
		}
	}
	list(rho=cvs[min.id],method=min.method)
}		

# Smith-Waterman  Algorithm
swpa2gc <- function(
			swdata = swd
			,id1
			,id2
			,cv.sim=0.95
			,sim="pearson"
			,method=2
			,anal=F
			,sp.pos=NA
			,opt=F
			,lm="sum"
			,vn=c("t1","t2")
	){
	# swdata: data
	# id1: index of target
	# id2: index of reference
	# cv.sim: cutoff value for similarity
	# sim: the method of similarity; dot and pearson
	# method: alignment method
	# 	=1
	#	=2
	#	=3
	#	=4
	# opt: optimality version with a certain method
	# sp.pos: information for the position of the spectra
	# vn: names for first and second retention times
	
	seq1 = swdata[[id1]]
	seq2 = swdata[[id2]]
	
	# scoring scheme
	MATCH    =  2 #1; # +1 for letters that match
	MISMATCH = -1 #-1; # -1 for letters that mismatch
	GAP      = -1; # -1 for any gap

	# number of samples
	len1 = dim(seq1)[1]
	len2 = dim(seq2)[1]

	# sp.pos
	if(is.na(sp.pos[1])){
		sp.pos = sort(grep("X",colnames(seq1)))
		sp.pos = sp.pos[1]:sp.pos[length(sp.pos)]
	}

	# index
	seq1$idx = 1:len1
	seq2$idx = 1:len2

	# initialization
	score.m = matrix(0,(len2+1),(len1+1))
	pointer.m = matrix("none",(len2+1),(len1+1))
	
	# fill
	max.i     = 0;
	max.j     = 0;
	max.score = 0;

	# similarity and distance
	mcor = sim.pd(seq2[,],seq1[,],ms.pos=sp.pos,sm=sim)
	mdist = dist.r(seq2[,vn],seq1[,vn],dm="euclidean")
	mdist = 1/(1+mdist)

	if(opt==F){
		s.align = sw4pa(
			para=cv.sim
			,tar=seq1
			,ref=seq2
			,method=method
			,out=2
			,sim=sim
			,sp.pos=sp.pos
			,vn = vn
		)
	}else{
		#ini = find.ini(sw4pa,tar=seq1,ref=seq2,method=method,out=1,sim=sim,sp.pos=sp.pos,vn=vn)
		#fit = nlminb(ini,sw4pa,lower=0,upper=1
		#		,tar=seq1,ref=seq2,method=method,out=1,sim=sim,sp.pos=sp.pos,vn=vn
		#	)
		#cv.sim = fit$par
		#cv.sim = ini
		opt = mult.shoot(sw4pa,tar=seq1,ref=seq2,method=method,out=1,sim=sim,lm=lm,sp.pos=sp.pos,vn=vn)
		cv.sim = opt$rho
		method = opt$method
		s.align = sw4pa(
			para=cv.sim
			,tar=seq1
			,ref=seq2
			,method=method
			,out=2
			,sim=sim
			,sp.pos=sp.pos
			,vn = vn
		)
	}
	#print(s.align)
	if(is.na(s.align)[1]){
		ll = rep(0,6)
		ll = as.data.frame(t(ll))
		dimnames(ll)[[2]] = c("ssim","sdist","psim","pdist","ss","sp")
		accur = as.data.frame(t(c(method,cv.sim,rep(-1,7))))
		dimnames(accur)[[2]] = c("method","rho","tpr","ppv","f1","tp","fp","tn","fn")
		idx = NA
		s.align = NA
	}else{	
		# alignment
		s.align = as.data.frame(s.align)
		dimnames(s.align)[[2]] = c("tt1","tt2","rt1","rt2","sim","dist","tid","rid")

		# likelihood
		#ll1 = sum(s.align$sim)
		#ll2 = sum(s.align$dist)
		#ll3 = -log(prod(s.align$sim))
		#ll4 = -log(prod(s.align$dist))
		#ll5 = ll1+ll2
		#ll6 = ll3+ll4
		#ll = c(ll1,ll2,ll3,ll4,ll5,ll6)
		#ll = as.data.frame(t(ll))
		#dimnames(ll)[[2]] = c("ssim","sdist","psim","pdist","ss","sp")
		
		if(lm=="sum"){
			ll = sum(s.align$sim)
		}else{
			ll = prod(s.align$sim)
		}
	
		# accuracy
		if(anal==T){
			accur = as.data.frame(t(c(method,cv.sim,cscore(seq1,seq2,s.align))))
			dimnames(accur)[[2]] = c("method","rho","tpr","ppv","f1","tp","fp","tn","fn")
		}else{
			accur = NA
		}
	
		# index	
		idx = as.data.frame(s.align[,c("tid","rid")])				
		dimnames(idx)[[2]] = c(as.character(id1),as.character(id2))
	}

	list(align=s.align,ll=ll,summary=accur,idx=idx,rho=cv.sim)
}

##########################################################
align <- function(padata,alist,method="peak",plot=F){
	# alist: it should be arranged.
	# e.g.: (1,2), (2,3), (3,4), (4,5), ...
	# Its order sould be consistent with that of padata
	# method: peak, name, area
	
	len = length(alist)
	wtable = alist[[1]]
	for(i in 2:len){
		#cat(dim(wtable)[2],i,"\n")
		ta = alist[[i]]
		#cat(colnames(ta)[1],colnames(alist[[i-1]])[2],"\n")
		if(colnames(ta)[1]!=colnames(alist[[i-1]])[2]){
			stop("!!! Please make sure the order of alist !!!\n")
		}
		t.pos = match(wtable[,i],ta[,1])
		wtable = data.frame(wtable[which(!is.na(t.pos)),],ta[na.omit(t.pos),2])
		dimnames(wtable)[[2]][i+1] = colnames(ta)[2]
		#print(wtable)
	}
	wlen = dim(wtable)[2]
	if(method=="peak"){
		rtable = list()
		rtable$t1 = matrix(0,dim(wtable)[1],dim(wtable)[2])
		rtable$t2 = rtable$t1
		for(i in 1:wlen){
			tmp.wt = wtable[,i]
			rtable$t1[,i] = padata[[i]]$t1[tmp.wt]
			rtable$t2[,i] = padata[[i]]$t2[tmp.wt]
		}
	}else if(method=="name"){
		rtable = matrix("",dim(wtable)[1],dim(wtable)[2])
		for(i in 1:wlen){
			tmp.wt = wtable[,i]
			rtable[,i] = as.character(padata[[i]]$name[tmp.wt])
		}
	}else if(method=="area"){
		rtable = matrix(0,dim(wtable)[1],dim(wtable)[2])
		for(i in 1:wlen){
			tmp.wt = wtable[,i]
			rtable[,i] = padata[[i]]$area[tmp.wt]
		}
	}
	if(plot){
		if(method=="peak"){
			plot(1:10,1:10,type="n",xlim=range(rtable$t1),ylim=range(rtable$t2),xlab="RT1",ylab="RT2")
			for(i in 1:wlen){
				points(rtable$t1[,i],rtable$t2[,i],col=i,pch=i)
			}
		}else if(method=="area"){
			plot(1:10,1:10,type="n",xlim=c(1,dim(rtable)[2]),ylim=range(rtable),xlab="Peak",ylab="Peak Area")
			for(i in 1:dim(rtable)[1]){
				points(1:wlen,rtable[i,],col=i,pch=i,lty=i,type="o")
			}
		}
	}
		
			
	list(pos=wtable,info=rtable)
}
			
comp.name <- function(ntable){
	nt = factor(ntable)
	len = dim(ntable)
	total = len[1]
	correct = 0
	for(i in 1:len[1]){	
		flag = 1
		for(j in 2:len[2]){
			if(ntable[i,j-1]!=ntable[i,j]){
				flag=0
				break
			}
		}
		if(flag==1){
			correct = correct + 1
		}
	}
	c(TP=correct,P=total,PPV=correct/total)
}
