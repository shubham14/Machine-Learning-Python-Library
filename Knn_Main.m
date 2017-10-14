function [Pred_Vector,Error]=Knn_Main(Train,Test,K)
    test_Label=Test(:,1);
    n=length(Test);
    Pred_vector=[];
    %for i=1:n
    %    Pred_vector=[Pred_vector ClassifyImage(Train,Test(i,:),K)];
    %end
    Pred_Vector=ClassifyImage(Train,Test,K);
    del=sum(Pred_vector==test_Label);
    Error=del/n;
end