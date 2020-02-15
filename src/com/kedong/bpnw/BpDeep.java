package com.kedong.bpnw;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;



public class BpDeep {
	public static double MAX=1*10E6;
	public static double MIN=-1*10E6;
	
	public Double [][]layer;//神经网络各层节点
	public Double [][]layerErr;//神经网络各节点误差
	public double [][][]layer_weight;//各节点权重，第一层的权重放在【1】
	public double [][][]layer_weight_delta;//各层节点权重动量，第一层的权重放在【1】
	
	public double [][]layer_moveB;//偏置量，第一层偏置为0
	public double [][]layer_moveB_delta;//偏置增量
	public double rate;//学习系数
	public double cost;//单次代价
	public int []layernum;//层数就是数组的长度，每层的节点数就是数组内数字
	/**
	 * 
	 * @param inData 需要进行归一化的输入数据{{2,3,4},{...},{...}},里面的任何一个{2,3,4}里面都是一组样本，维度为3
	 * @param outData 归一化完毕的数据{{},{},{}},
	 * @param norm 归一化过程的最大值最小值{{min,max},{min,max}}，norm的长度为inData里面一个样本数据的维度数例子为3
	 */
	public void normalizeSample(Double inData[][],Double outData[][],Double norm[][]) {
//		double []in_norm;//输入归一化因数
//		norm = new double[inData[0].length][2];
		//初始化归一化所需要的参数
		for(int i=0;i<norm.length;i++) {
			norm[i][0]=MAX;
			norm[i][1]=MIN;
		}
		
		for(int i=0;i<inData.length;i++) {
			for(int j=0;j<inData[i].length;j++) {
				if(inData[i][j]>norm[j][1])norm[j][1]=inData[i][j];
				if(inData[i][j]<norm[j][0])norm[j][0]=inData[i][j];
			}
		}
//		outData = inData.clone();
		for(int i=0;i<inData.length;i++) {
			for(int j=0;j<inData[i].length;j++) {
				outData[i][j] = (inData[i][j]-norm[j][0])/(norm[j][1]-norm[j][0]);
			}
		}
	}
	
	/**
	 * 
	 * @param inData 需要进行归一化的输入数据{{2,3,4},{...},{...}},里面的任何一个{2,3,4}里面都是一组样本，维度为3
	 * @param outData 归一化完毕的数据{{},{},{}},
	 * @param norm 归一化过程的最大值最小值{{min,max},{min,max}}，norm的长度为inData里面一个样本数据的维度数例子为3
	 */
	public void normalizeIn(Double inData[],Double outData[],Double norm[][]) {

		for(int j=0;j<inData.length;j++) {
			outData[j] = (inData[j]-norm[j][0])/(norm[j][1]-norm[j][0]);
		}
	}
	
	
	
	public static void main(String[] args) {
		double devide = 1*10e-12;
		Double[][]data = new Double[][] {{1d,2d},{2d,2d},{1d,1d},{2d,1d},{3d,2d},{1.5,1.7},{1.1,1.2},{0.9,1.4},{0.8,1.6},{1.6,1.5}};
		Double [][]target= new Double[][] {{3d},{10d},{2d},{9d},{29d},{5.075},{2.531},{2.129},{2.112},{5.596}};
		Double [][]in_norm;//输入归一化因数
		Double [][]out_norm;//输出归一化因数
		in_norm = new Double[data[0].length][2];
		out_norm = new Double[target[0].length][2];

		Double[][]dataNorm = BpDeepAllSample.copyDeep2(data);
		Double [][]targetNorm =  BpDeepAllSample.copyDeep2(target);
		
		BpDeep bp = new BpDeep();
		bp.normalizeSample(data,dataNorm,in_norm);
		bp.normalizeSample(target,targetNorm,out_norm);
		
		
		bp.init(new int [] {2,5,1},0.2);
		long count = 0;
		while(true) {
			double sub = 0;
			for(int i=0;i<data.length;i++) {
				Double[] out = bp.computOut(dataNorm[i]);
				bp.updateWeight(targetNorm[i]);
				sub = 0;
				for(int j=0;j<out.length;j++) {
					sub+=Math.pow(out[j]-targetNorm[i][j],2);
				}
				sub = sub/out.length;
				System.out.println("第"+count+"次迭代,误差："+sub+",输入:"+Arrays.toString(data[i])+",输出："+Arrays.toString(target[i]));
				System.out.println("输出："+out[0]+",target:"+target[i][0]);
				count++;
				if(sub<devide)break;
			}
			if(sub<devide)break;
			sub=0;
		}
		Double dataTestIn[]=new Double[]{2d,1.5};
		Double dataTestNorm[]=new Double[2];
		bp.normalizeIn(dataTestIn, dataTestNorm, in_norm);
		Double []out = bp.computOut(dataTestNorm);
		out[0] = out[0]*(out_norm[0][1]-out_norm[0][0])+out_norm[0][0];
		System.out.println(Arrays.toString(out));
	}
	public BpDeep() {
		
	}
	@Test
	public void testRandom21() {
		Random random = new Random();
		for(int i=0;i<20;i++)
			System.out.println(random.nextDouble());
	}
	
	/**
	 * 
	 * @param layernum 每层神经元节点数目
	 * @param rate 学习系数
	 */
	public void init(int []layernum,double rate){
		this.rate = rate;
		this.layernum = layernum.clone();
//		this.layernum = new int[layernum.length];
//		for(int i=0;i<layernum.length;i++)
		
		layer = new Double[layernum.length][];
		layerErr = new Double[layernum.length][];
		
		layer_weight = new double[layernum.length][][];
		layer_weight_delta = new double[layernum.length][][];
		
		layer_moveB =  new double[layernum.length][];
		layer_moveB_delta =  new double[layernum.length][];
		
		
		
		Random random = new Random();
		//检查过了，layer_weight_delta里面每层的维度有点异议
		for(int l=0;l<layernum.length;l++) {
			layer[l] = new Double[layernum[l]];
			layerErr[l] = new Double[layernum[l]];
			
			if(l>0) {
				layer_weight[l] = new double[layernum[l]][layernum[l-1]];//
				layer_weight_delta[l] = new double[layernum[l]][layernum[l-1]];
				layer_moveB[l] = new double [layernum[l]];
				layer_moveB_delta[l] = new double [layernum[l]];
				
				for(int j=0;j<layernum[l];j++) {//weight下标从1开始
					layer_moveB[l][j] = random.nextDouble();
					for(int i=0;i<layernum[l-1];i++) {
						layer_weight[l][j][i] = random.nextDouble();
					}
				}
			}
		
		}
		
	}
	
	
	/**
	 * 
	 * weight矩阵的下标是从1开始的，一直到layernum-1，如果layernum=5，那么weight下标0,1,2,3,4，其中0中的数不用
	 * @param in
	 * @return
	 */
	public Double[]computOut(Double[]in){
		for(int l=1;l<layer.length;l++) {
			for(int j=0;j<layer[l].length;j++) {
				double z= layer_moveB[l][j];//初始化为b
				for(int i=0;i<layer[l-1].length;i++) {
					layer[l-1][i]=l==1?in[i]:layer[l-1][i];
					z+=layer_weight[l][j][i]*layer[l-1][i];
				}
				layer[l][j] = 1/(1+Math.exp(-z));
			}
		}
		return layer[layer.length-1];
	}
	
	/**
	 * 
	 * @param tar
	 */
	public void updateWeight(Double[]tar) {
		int l=layer.length-1;
		for(int j=0;j<layernum[l];j++)
			layerErr[l][j]=-layer[l][j]*(1-layer[l][j])*(tar[j]-layer[l][j]);//
		
		while(l>0) {
			for(int j=0;j<layernum[l];j++) {
				for(int i=0;i<layernum[l-1];i++) {
					double dltW_ji = layerErr[l][j]*layer[l-1][i];
					layer_weight[l][j][i] = layer_weight[l][j][i]-rate*dltW_ji;
					layer_moveB[l][j] = layer_moveB[l][j]-rate*layerErr[l][j];
				}
				
			}
			l--;
			
			for(int j=0;j<layerErr[l].length;j++) {
				double sita_l_j = 0;
				for(int k=0;k<layernum[l+1];k++) {
					sita_l_j += layer_weight[l+1][k][j]*layerErr[l+1][k]*layer[l][j]*(1-layer[l][j]);
				}
				layerErr[l][j] = sita_l_j;
			}
		}

	}
}
