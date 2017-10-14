function Pred_Classes=ClassifyImage(Train,Test,K)
    y=Train(:,1);  %Class labels
    Tr=Train(:,2:end);
    Ts=[];
    Ts_label=[];
    TsSample=randsample(10001,100);
    for i=1:length(TsSample)
        TsSample1=Test(TsSample(i),:);
        Ts=[Ts;TsSample1(2:end)]; %stores normed subtracted results
        Ts_label=[Ts_label;TsSample1(:,1)];
    end
    %Ts1=Ts(1:100);
    Pred_Classes=[];
    %S=bsxfun(@minus,Train(2,:),Test);       %subtracting the indivisual elements
    count=0;
    countErr=0;
    for i=1:101
        D2=[];
        S=bsxfun(@minus,Tr,Ts(i,:));
        D2=[D2;arrayfun(@(n) norm(S(n,:),1), 1:size(S))];
        D2=[D2;y'];
        DistArray=sortrows(D2',1);
        D1=DistArray(1:K,2);
        Class=mode(D1)
        if Class ~= Ts_label(i)
            countErr=countErr+1
        end 
        Pred_Classes=[Pred_Classes;Class];
        count=count+1
    end
end