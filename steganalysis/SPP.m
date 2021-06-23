function F = SPP(FCB)
% -------------------------------------------------------------------------
% 2019.1.7
% 
% -------------------------------------------------------------------------
%
% -------------------------------------------------------------------------
% Input:  FCB ... path to the FCB
%
% Output: f ....... extracted SPP features
% -------------------------------------------------------------------------

	matrix1 = single(FCB);
	matrix = matrix1;
	matrix = floor(matrix/5);
	
	F = zeros(1,7);

	track_0=matrix(:,[1,6]);
	track_1=matrix(:,[2,7]);
	track_2=matrix(:,[3,8]);
	track_3=matrix(:,[4,9]);
	track_4=matrix(:,[5,10]);
    feature = [];
	feature = [feature;getSPP(track_0)];
	feature = [feature;getSPP(track_1)];
	feature = [feature;getSPP(track_2)];
	feature = [feature;getSPP(track_3)];
	feature = [feature;getSPP(track_4)];

	F = double( mean( feature)); 



function f = getSPP(track)
     
    A1 = track(:,1);
    A2 = track(:,2);
	
f = zeros(1,7);


	for i = 1 : 7
        A2_temp = A2(A1(:)==i);            
        f(i) = sum(A2_temp(:)==i);             	
    end
	

	 f = double(f);
	 N_subrame = size(track,1); 	
	 if N_subrame>0, f = f/N_subrame;  end
		
