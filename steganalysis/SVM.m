clear all;
clc;
ACC1=[];

load F_SPP_0_block.mat
F_SPP0=F_SPP;
label1 = [ones(length(F_SPP0),1)];

load F_SPP_1_block.mat
F_SPP1=F_SPP;
label2 = [2*ones(length(F_SPP1),1)];
%%%%
SPPtrain=[F_SPP0;F_SPP1];
LABELtrain=[label1;label2];
options.MaxIter = 10000000;
svmStruct = svmtrain(SPPtrain,LABELtrain,'Options', options);

%%% test%%%%
CLASS2=[];
CM=[];
 Y=F_SPP0;
d = floor(length(Y)/5);
     for  k=1:size(Y)/d
         %ENRGY2=ENRGY0(1:(V(2)-V(1))*length(ENRGY0)/100,:);
         F_SPP2=F_SPP0((d*(k-1)+1:d*k),:);
         class1=svmclassify(svmStruct,F_SPP2);
         labeltest=label1((d*(k-1)+1:d*k),:);
         cm=confusionmat(class1,labeltest);
         CLASS2=[CLASS2  class1];
         CM=[CM cm];
         class1=[];
         cm=[];
     end
     %%%%% accuracy1 %%%%
     
     for h=1:5
         
 YY=accumarray(CLASS2(:,h), 1);
         ACC=YY(2)/(YY(1)+YY(2));
         ACC1=[ACC1 ACC];
         ACC=[];
     end
     moy=mean(ACC1);
     TER=1-moy
