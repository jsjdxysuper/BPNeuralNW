package com.kedong.bpnw;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;



public class BpDeepAllSample {
	public static double MAX=1*10E6;
	public static double MIN=-1*10E6;
	public double devide = 1*10e-5;
	public Double[][]data;
	public Double [][]target;
	
	public Double [][]layer;//���������ڵ�
	public Double [][]layerErr;//��������ڵ����
	public Double [][][]layer_weight;//���ڵ�Ȩ�أ���һ���Ȩ�ط��ڡ�1��
	public Double [][][]layer_weight_delta;//����ڵ�Ȩ�ض�������һ���Ȩ�ط��ڡ�1��
	
	public Double [][]layer_moveB;//ƫ��������һ��ƫ��Ϊ0
	public Double [][]layer_moveB_delta;//ƫ������
	public double rate;//ѧϰϵ��
	public double cost;//���δ���
	public int []layernum;//������������ĳ��ȣ�ÿ��Ľڵ�����������������
	
	
	
	
	public Double [][][][]every_layer_weight_delta;//����ڵ�Ȩ�ض�������һ���Ȩ�ط��ڡ�1������һ����������ڼ����������ڶ���Ϊ��������������Ϊ��������ڵ����
	public Double [][][]every_layer_moveB_delta;//ƫ������
	public Double [][]output;//���е������������һ������Ϊ������ţ��ڶ�����i��Ϊ������i�������
	/**
	 * 
	 * @param inData ��Ҫ���й�һ������������{{2,3,4},{...},{...}},������κ�һ��{2,3,4}���涼��һ��������ά��Ϊ3
	 * @param outData ��һ����ϵ�����{{},{},{}},
	 * @param norm ��һ�����̵����ֵ��Сֵ{{min,max},{min,max}}��norm�ĳ���ΪinData����һ���������ݵ�ά��������Ϊ3
	 */
	public void normalizeSample(Double inData[][],Double outData[][],Double norm[][]) {
//		double []in_norm;//�����һ������
//		norm = new double[inData[0].length][2];
		//��ʼ����һ������Ҫ�Ĳ���
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
	 * @param inData ��Ҫ���й�һ������������{{2,3,4},{...},{...}},������κ�һ��{2,3,4}���涼��һ��������ά��Ϊ3
	 * @param outData ��һ����ϵ�����{{},{},{}},
	 * @param norm ��һ�����̵����ֵ��Сֵ{{min,max},{min,max}}��norm�ĳ���ΪinData����һ���������ݵ�ά��������Ϊ3
	 */
	public void normalizeIn(Double inData[],Double outData[],Double norm[][]) {

		for(int j=0;j<inData.length;j++) {
			outData[j] = (inData[j]-norm[j][0])/(norm[j][1]-norm[j][0]);
		}
	}
	
	/**
	 * ��¼ĳ��������Ȩ��ƫ��
	 * @param sampleIndex ��������
	 */
	public void recordDelta(int sampleIndex) {
		for(int l=layer.length-1;l>0;l--) {
			for(int j=0;j<layernum[l];j++) {
				for(int i=0;i<layernum[l-1];i++) {
					every_layer_weight_delta[sampleIndex][l][j][i] = layer_weight_delta[l][j][i];
					every_layer_moveB_delta[sampleIndex][l][j] = layer_moveB_delta[l][j];
				}
				
			}
		}
	}
	/**
	 * 
	 * @param sampleIndex
	 */
	public void recordOutput(int sampleIndex,Double oneSampleOut[]) {
//		output
		for(int i=0;i<oneSampleOut.length;i++) {
			output[sampleIndex][i] = oneSampleOut[i];
		}
	}
	
	public static void main(String[] args) {
		BpDeepAllSample bp = new BpDeepAllSample();
		
		bp.data = new Double[][] {{1d,2d},{2d,2d},{1d,1d},{2d,1d},{3d,2d},{1.5,1.7},{1.1,1.2},{0.9,1.4},{0.8,1.6},{1.6,1.5}};
//		bp.target= new Double[][] {{3d},{4d},{2d},{3d},{5d},{3.2},{2.3},{2.3},{2.4},{3.1}};
		bp.target= new Double[][] {{3d},{4d},{2d},{3d},{5d},{3.2},{2.3},{2.3},{2.4},{3.1}};
		Double [][]in_norm;//�����һ������
		Double [][]out_norm;//�����һ������
		in_norm = new Double[bp.data[0].length][2];
		out_norm = new Double[bp.target[0].length][2];

		Double[][]dataNorm = bp.copyDeep2(bp.data);//��һ�������������
		Double [][]targetNorm = bp.copyDeep2(bp.target);//��һ�����Ŀ������
		
		
		bp.normalizeSample(bp.data,dataNorm,in_norm);
		bp.normalizeSample(bp.target,targetNorm,out_norm);
		
		
		bp.init(new int [] {2,5,1},0.5);
		long count = 0;
		double totalError = 0;
		bp.initAllSampleDelta(bp.data,bp.target);
		while(true) {
			count++;
			for(int i=0;i<bp.data.length;i++) {
				Double[] out = bp.computOut(dataNorm[i]);
				
				bp.updateWeight(targetNorm[i]);
				
				bp.recordDelta(i);
				bp.recordOutput(i,out);
			}
			totalError=0;
//			System.out.println(count+":target,output");
			for(int i=0;i<bp.data.length;i++) {
				
				for(int j=0;j<bp.target[i].length;j++) {
					totalError += Math.pow(targetNorm[i][j]-bp.output[i][j], 2);
//					System.out.println(targetNorm[i][j]+","+bp.output[i][j]);
				}
			}
			totalError = totalError/bp.data.length;
			System.out.println("��"+count+"�ε������������"+totalError);
			if(totalError<bp.devide) {//����Ҫ���ˣ�����Ҫ������
				break;
			}else {
				Double [][][]this_layer_weight_delta = copyDeep3(bp.layer_weight_delta);//����ڵ�Ȩ�ض�������һ���Ȩ�ط��ڡ�1��
				initArray3(this_layer_weight_delta, 0d);
				Double [][]this_layer_moveB_delta = copyDeep2(bp.layer_moveB_delta);//ƫ������
				initArray2(this_layer_moveB_delta, 0d);
				for(int i=0;i<bp.data.length;i++) {
					for(int j=1;j<bp.every_layer_weight_delta[i].length;j++) {
						for(int k=0;k<bp.every_layer_weight_delta[i][j].length;k++) {
							for(int l=0;l<bp.every_layer_weight_delta[i][j][k].length;l++) {
								this_layer_weight_delta[j][k][l]+=bp.every_layer_weight_delta[i][j][k][l];
							}
						}
					}
				}
				
				
				
				for(int i=1;i<bp.layer_weight.length;i++) {
					for(int j=0;j<bp.layer_weight[i].length;j++) {
						for(int k=0;k<bp.layer_weight[i][j].length;k++) {
							bp.layer_weight[i][j][k]-=bp.rate*this_layer_weight_delta[i][j][k]/bp.data.length;
						}
					}
				}
				for(int i=0;i<bp.data.length;i++) {
					for(int j=1;j<bp.every_layer_moveB_delta[i].length;j++) {
						for(int k=0;k<bp.every_layer_moveB_delta[i][j].length;k++) {
							this_layer_moveB_delta[j][k] += bp.every_layer_moveB_delta[i][j][k];
						}
					}
				}
				for(int i=1;i<bp.layer_moveB.length;i++) {
					for(int j=0;j<bp.layer_moveB[i].length;j++) {
						bp.layer_moveB[i][j] -= bp.rate*this_layer_moveB_delta[i][j]/bp.data.length;
					}
				}
				
				
			}
//			sub = 0;
//			for(int j=0;j<out.length;j++) {
//				sub+=Math.pow(out[j]-targetNorm[i][j],2);
//			}
//			sub = sub/out.length;
//			System.out.println("��"+count+"�ε���,��"+sub+",����:"+Arrays.toString(data[i])+",�����"+Arrays.toString(target[i]));
//			System.out.println("�����"+out[0]+",target:"+target[i][0]);
//			count++;
//			if(sub<devide)break;
//			if(sub<devide)break;
//			sub=0;
		}
		Double dataTestIn[]=new Double[]{1.1d,1.5};
		Double dataTestNorm[]=new Double[2];
		bp.normalizeIn(dataTestIn, dataTestNorm, in_norm);
		Double []out = bp.computOut(dataTestNorm);
		out[0] = out[0]*(out_norm[0][1]-out_norm[0][0])+out_norm[0][0];
		System.out.println(Arrays.toString(out));
	}
	public BpDeepAllSample() {
		
	}
	@Test
	public void testRandom21() {
		Random random = new Random();
		for(int i=0;i<20;i++)
			System.out.println(random.nextDouble());
	}
	
	/**
	 * ��ʼ������ƫ��
	 * @param data
	 */
	public void initAllSampleDelta(Double[][]data,Double [][]target) {
		every_layer_weight_delta = new Double[data.length][][][];
		every_layer_moveB_delta = new Double [data.length][][];
		output = new Double[data.length][];
		for(int i=0;i<data.length;i++) {
			every_layer_weight_delta[i] = copyDeep3(layer_weight_delta);
			initArray3(every_layer_weight_delta[i], 0d);
			every_layer_moveB_delta[i] = copyDeep2(layer_moveB_delta);
			initArray2(every_layer_moveB_delta[i], 0d);
			output[i] = new Double[target[i].length];
			
		}
	}
	/**
	 * 
	 * @param layernum ÿ����Ԫ�ڵ���Ŀ
	 * @param rate ѧϰϵ��
	 */
	public void init(int []layernum,double rate){
		this.rate = rate;
		this.layernum = layernum.clone();
//		this.layernum = new int[layernum.length];
//		for(int i=0;i<layernum.length;i++)
		
		layer = new Double[layernum.length][];
		layerErr = new Double[layernum.length][];
		
		layer_weight = new Double[layernum.length][][];
		layer_weight_delta = new Double[layernum.length][][];
		
		layer_moveB =  new Double[layernum.length][];
		layer_moveB_delta =  new Double[layernum.length][];
		
		
		
		Random random = new Random();
		//�����ˣ�layer_weight_delta����ÿ���ά���е�����
		for(int l=0;l<layernum.length;l++) {
			layer[l] = new Double[layernum[l]];
			layerErr[l] = new Double[layernum[l]];
			
			if(l>0) {
				layer_weight[l] = new Double[layernum[l]][layernum[l-1]];//
				layer_weight_delta[l] = new Double[layernum[l]][layernum[l-1]];
				layer_moveB[l] = new Double [layernum[l]];
				layer_moveB_delta[l] = new Double [layernum[l]];
				
				for(int j=0;j<layernum[l];j++) {//weight�±��1��ʼ
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
	 * weight������±��Ǵ�1��ʼ�ģ�һֱ��layernum-1�����layernum=5����ôweight�±�0,1,2,3,4������0�е�������
	 * @param in
	 * @return
	 */
	public Double[]computOut(Double[]in){
		for(int l=1;l<layer.length;l++) {
			for(int j=0;j<layer[l].length;j++) {
				double z= layer_moveB[l][j];//��ʼ��Ϊb
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
	 * @param tar ���������������
	 */
	public void updateWeight(Double[]tar) {
		int l=layer.length-1;
		for(int j=0;j<layernum[l];j++)
			layerErr[l][j]=-layer[l][j]*(1-layer[l][j])*(tar[j]-layer[l][j]);//
		
		while(l>0) {
			for(int j=0;j<layernum[l];j++) {
				for(int i=0;i<layernum[l-1];i++) {
					double dltW_ji = layerErr[l][j]*layer[l-1][i];
					layer_weight_delta[l][j][i] = dltW_ji;
//					layer_weight[l][j][i] = layer_weight[l][j][i]-rate*dltW_ji;
					layer_moveB_delta[l][j] = layerErr[l][j];
//					layer_moveB[l][j] = layer_moveB[l][j]-rate*layerErr[l][j];
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
	

	/**
	 * ��ά�������
	 * @param a
	 * @param b
	 */
	public static Double [][][] copyDeep3(Double [][][]a) {
		Double [][][]b = new Double[a.length][][];
		for(int i=0;i<a.length;i++) {
			if(a[i]!=null) {
				b[i] = new Double[a[i].length][];
				for(int j=0;j<a[i].length;j++) {
					b[i][j] = new Double[a[i][j].length];
					for(int k=0;k<a[i][j].length;k++) {
						b[i][j][k]=a[i][j][k];
					}
				}
			}
		}
		return b;
	}
	
	/**
	 * ��ά�������
	 * @param a
	 * @param b
	 */
	public static Double[][] copyDeep2(Double [][]a) {
		Double [][]b = new Double[a.length][];
		for(int i=0;i<a.length;i++) {
			if(a[i]!=null) {
				b[i] = new Double[a[i].length];
				for(int j=0;j<a[i].length;j++) {
					b[i][j]=a[i][j];
				}
			}
		}
		return b;
	}
	
	public static <E>void initArray2(E [][]a,E val) {
		for(int i=0;i<a.length;i++) {
			if(a[i]!=null) {
				for(int j=0;j<a[i].length;j++) {
					a[i][j] = val;
				}
			}
		}
	}
	
	public static <E>void initArray3(E [][][]a,E val) {
		for(int i=0;i<a.length;i++) {
			if(a[i]!=null) {
				for(int j=0;j<a[i].length;j++) {
					for(int k=0;k<a[i][j].length;k++) {
						a[i][j][k] = val;
					}
				}
			}
		}
	}
	
	@Test
	public void test() {
		Double a[][]= {{1d,2d,3d},{4d,5d},{6d,7d,8d,9d}};
		initArray2(a,0d);
		System.out.println();
	}
	@Test
	public void test1() {
		double[][]ab = new double[4][];
		ab[0] = new double[2];
		ab[1] = new double[3];
		ab[2] = new double[4];
		System.out.println();
	}
	@Test
	public void testDeepCopy2() {
		Double a[][]= {{1d,2d,3d},{4d,5d},{6d,7d,8d,9d}};
		Double b[][]= {};
		b=copyDeep2(a);
		b[0][2]=21d;
		System.out.println();
	}
	@Test
	public void testDeepCopy3() {
		Double a[][][]= {{{1d,2d},{1d,2d,3d}},{{1d},{2d,3d,4d},{5d,6d}}};
		Double b[][][]= {};
		b=copyDeep3(a);
		b[0][1][2]=21d;
		System.out.println();
	}
}
