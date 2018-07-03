library(ggplot2)
library(data.table)
library(dplyr)
library(reshape2)
library(stringr)
setwd("C:/Users/mmnet/Desktop/DLCV2018SPRING/final/")

ext = ".png"
save_dir = "plot/"

d1 <- fread("save/our_model_shape.csv")
dc <- fread("save/VGG_model_shape.csv")
dc <- dc[4,-1]
d1 <- d1[4,-1]
for(i in seq(1,ncol(dc))){
  dc[1,i] = dc[1,i,with=F]-d1[1,i,with=F]
}

d1 <- rbind(d1,dc)
d1$type = factor(c("Kept","Pruned"),levels=c("Pruned","Kept"))
pd1 <- melt(d1)
ggplot(pd1, aes(x=variable, y=value, fill=type))+
  geom_bar(stat='identity')+
  scale_y_continuous(breaks=c(0,64,128, 256, 512))+
  #coord_flip()+
  scale_fill_manual(values=c("#d9d9d9","#80b1d3"))+
  theme_bw()+
  ylab("Channel index")+
  xlab("")+
  ggtitle("Compressed Network")+
  theme(legend.position = c(0.1,0.8),
        plot.title =element_text(size=28,hjust = 0.5),
        legend.title = element_text(size=24),
        legend.text = element_text(size=24),
        axis.title = element_text(size=20),
        axis.text = element_text(size=20),
        legend.key.size=unit(2,"line"))
ggsave(filename=paste0(save_dir,"CP_pruned_net",ext),width = 12, height=6, units='in')

tf <- c("save/gamma.csv")

p <- list()
for(i in seq_along(tf)){
  value <- c()
  d <- fread(tf[i], header = F)
  dd <- strsplit(d$V2, ",")
  names(dd) <- d$V1
  value <- c(as.numeric(dd$conv1_1_gamma),
             as.numeric(dd$conv1_2_gamma),
             as.numeric(dd$conv2_1_gamma),
             as.numeric(dd$conv3_1_gamma),
             as.numeric(dd$conv4_1_gamma),
             as.numeric(dd$conv5_3_gamma))
  
  target <- c("conv1_1_gamma","conv5_3_gamma")
  mycolor = c("#fdb462","#80b1d3","#b3de69","#fb8072","#bc80bd", "gray")
  for(i in seq_along(target)){
    tar = target[i]
    col = mycolor[i]
    print(dd[[target[i]]])
    pp = data.table("index"= seq(1, length(dd[[target[i]]])),
                    "value"= as.numeric(dd[[target[i]]]),
                    "col" = rep(col, length(dd[[target[i]]])))
    ggplot(data=pp,aes(x = index, y=value, group=1, colour=col))+
      geom_line()+
      geom_point()+
      xlab("Channel index")+
      ylab("Value of scaling factors")+
      ggtitle(target[i])+
      theme_bw()+
      theme(plot.title =element_text(size=28,hjust = 0.5),
            axis.title = element_text(size=20),
            axis.text = element_text(size=20),
            legend.position="none")
    ggsave(filename=paste0(save_dir,target[i],ext),width = 6, height=4, units='in')
  } 
}

dfx <- fread("save/summary.csv", data.table = F)
dfx$color <- c("gray","#80b1d3","#b3de69","#fb8072","#bc80bd", "#fdb462")
with(dfx, symbols(x=`Multi-Adds (G)`, y=`Accuary (%)`, circles=`Parameters (M)`,
                  main = "Accuracy vs. Resource Requirements",
                  inches=1,ylim = c(55,100),xlim=c(0,30),ann=T, bg=color, fg=NULL,
                  width = 6, height = 4, units = 'in'))
text(dfx$`Multi-Adds (G)`, dfx$`Accuary (%)`, labels = dfx$method)
