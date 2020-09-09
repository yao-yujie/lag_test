#include<stdio.h>
/*
spec[]是输入的光谱，e_lo[]能量小，e_hi[]能量大，
retur[]返回值，lspec 光谱的长度，
nn 合并数，e_n retur的长度
gcc -std=c99 specturn_tool.c -shared -o spectrum_tool.so.6

*/

void A_spec(double spec[],double e_lo[] ,double e_hi[] ,double retur[],int lspec,int nn,int e_n);



void A_spec(double spec[],double e_lo[] ,double e_hi[] ,double retur[],int lspec,int nn,int e_n)
{
	int i,j,k;
	float sum;

	if(lspec%nn == 0){
		//printf("start %p\n",retur);
		//printf("if %d\n",lspec%nn);
		sum = 0.;j = 0;k=0;
		for(i=0;i<lspec;i++)
		{	
			sum += spec[i];
			j+=1;
			if (j >= nn){
				retur[k] = (sum/nn)*(e_hi[k]-e_lo[k]);
				//printf("%f\n",retur[k]);
				sum = 0.;
				j = 0;
				k++;
				}
		}
		//printf("stop %p\n",retur);
		//for(i=0;i<e_n;i++){printf("%f\n",retur[i]);}
		}
	else{
		//printf("else %d\n",lspec%nn);
		for(i=0;i<e_n;i++)
		{
			retur[i] = 0.;
		}}
	//return retur;
}







