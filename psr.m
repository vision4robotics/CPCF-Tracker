function psr = psr(response,rate)
%maximum
maxresponse = max(response(:));
%The main peak window
range = ceil(sqrt(numel(response))*rate/2);
%The side lobe area
[xx, yy] = find(response == maxresponse, 1);
idx = xx-range:xx+range;
idy = yy-range:yy+range;
idy(idy<1)=1;idx(idx<1)=1;
idy(idy>size(response,2))=size(response,2);idx(idx>size(response,1))=size(response,1);
central_area = response(idx,idy);

response(idx,idy)=0;
lobe_num = numel(find(response~=0));
mean_value = sum(sum(response))/lobe_num;

response  = response - mean_value;
response(idx,idy)=0;

std_devia = ( sum(  sum(response.^2)  )  /  lobe_num )^0.5;


num = maxresponse - mean_value;
den = std_devia;
psr = num / den;
end