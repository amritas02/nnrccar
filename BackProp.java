

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions; 
import org.jblas.util.*;
import java.lang.Math;

public class BackProp 
{
	public static DoubleMatrix nnCostFunction(DoubleMatrix Theta1,DoubleMatrix Theta2, int input_layer_size, int output_layer_size, int num_labels, DoubleMatrix X, DoubleMatrix y, int lambda)
	{
		//reshaping parameters useless. No need to unroll parameters from calling functiom.
		
		int m = X.getRows(); 
		double J = 0.0; 
		
		DoubleMatrix Theta1_grad = new DoubleMatrix(Theta1.getRows(), Theta1.getColumns());
		DoubleMatrix Theta2_grad = new DoubleMatrix(Theta2.getRows(), Theta2.getColumns());
		
		X = addOnes(X);
		DoubleMatrix z2 = sigmoid(X.mmul(Theta1.transpose()));
		z2 = addOnes(z2);
		DoubleMatrix z3 = sigmoid(z2.mmul(Theta2.transpose()));
		
		DoubleMatrix y2 = new DoubleMatrix(num_labels, m);
		
		for(int i = 0; i<m ; i++)
		{
			double k = y2.get(i, 0);
			y2.put((int) (k-1), i, 1);
			
		}
		
		//J = ( sum ( sum ( (-1*y2) .* log(z3') - (1 - y2) .* log(1 - z3') ) ) ) / m ;
		
		//1-z3'
		DoubleMatrix r1 = new DoubleMatrix(); 
		z3.transpose().rsubi(1.0, r1);
		
		DoubleMatrix r2 = MatrixFunctions.log(r1);
		
		DoubleMatrix r3 = new DoubleMatrix(); 
		y2.rsubi(1.0, r3);
		
		DoubleMatrix r4 = r3.mul(r2);
		
		DoubleMatrix temp = y2.dup(); 
		
		DoubleMatrix r5 = (temp.neg()).mul(MatrixFunctions.log(z3.transpose()));
		
		double s = r5.sum(); 
		
		s = s/(double)m;
		J = s; 
		
		DoubleMatrix t1 = Theta1.dup(); 
		DoubleMatrix t2 = Theta2.dup(); 
		
		Theta1 = delColumn(Theta1);
		Theta2 = delColumn(Theta2);
		
		double temp2 = (lambda*(MatrixFunctions.powi(Theta1.transpose(),2).sum() + MatrixFunctions.powi(Theta2,2).sum() ))/(double)(2*m);
		
		J = J + temp2;
		
		DoubleMatrix DELTA1, DELTA2; 
		
		DELTA1 = DoubleMatrix.zeros(t1.getRows(), t1.getColumns());
		DELTA2 = DoubleMatrix.zeros(t2.getRows(), t2.getColumns());
		
		for( int t =0; t<m; t++)
		{
			DoubleMatrix a1 = X.getRow(t);
			DoubleMatrix Z2 = t1.mmul(a1.transpose());
			DoubleMatrix A2 = sigmoid(Z2);
			A2 = addRow(A2);
			DoubleMatrix A3 = sigmoid(t2.mmul(A2));
			DoubleMatrix delta3 = A3.sub(y2.getColumn(t));
			Z2 = addRow(Z2);
			DoubleMatrix delta2 = t2.transpose().mmul(delta3).mul(sigmoidGradient(Z2));
			delta2 = delta2.getRange(0,delta2.getRows(), 1, delta2.getColumns());
			
			
			DELTA2 = DELTA2.add(delta3.mmul(A2.transpose()));
			DELTA1 = DELTA1.add(delta2.mmul(a1));
		}
		
		
		DoubleMatrix aux1, aux2; 
		aux1 = addZeros(Theta1); 
		aux2 = addZeros(Theta2);
		
		Theta1_grad = DELTA1.add(aux1.mul(lambda)).divi(m);
		Theta2_grad = DELTA2.add(aux2.mul(lambda)).divi(m);
	
		//have to return a matrix with [J ; Theta1_grad ; Theta2_grad ]
		DoubleMatrix J2 = new DoubleMatrix(1,1);
		J2.put(0,0, J);
		
		DoubleMatrix costFunc; 
		
		Theta1_grad.reshape(Theta1_grad.getRows()*Theta1_grad.getColumns(), 1);
		Theta2_grad.reshape(Theta2_grad.getRows()*Theta2_grad.getColumns(), 1);
		
		DoubleMatrix temp3 = DoubleMatrix.concatVertically(J2, Theta1_grad);
		costFunc = DoubleMatrix.concatVertically(temp3, Theta2_grad);
		return costFunc; 
	}
	
	
	
	
	public static DoubleMatrix sigmoidGradient(DoubleMatrix z)
	{
		DoubleMatrix g = DoubleMatrix.zeros(z.getRows(), z.getColumns());
		DoubleMatrix r1 = new DoubleMatrix();
		sigmoid(z).rsubi(1.0, r1);
		g = sigmoid(z).mul(r1);
		
		return g;
		
	}

	public static DoubleMatrix predict (DoubleMatrix Theta1, DoubleMatrix Theta2, DoubleMatrix X)
	{
	  int m = X.getRows();
	  int n = X.getColumns(); 
	  
	  int num_labels = Theta2.getRows(); 
	  
	  DoubleMatrix p = DoubleMatrix.zeros(m,1);
	  
	  DoubleMatrix t1 = Theta1.transpose(); 
	   DoubleMatrix t2 = Theta2.transpose(); 
		
	    DoubleMatrix x = addOnes(X);
	  
		
		DoubleMatrix h1 = sigmoid(x.mmul(t1));
		h1 = addOnes(h1);
		DoubleMatrix h2 = sigmoid(h1.mmul(t2));
		
		//[dummy, p] = max(h2, [], 2);
		//store max element from the columns of h2 into p
		
		for(int i =0; i<h2.getRows();i++)
		{
			double max = 0.0;
			for(int j=0; j<h2.getColumns();j++)
			{
				if(h2.get(i, j) > max)
					max = h2.get(i, j);
			}
			p.put(i, 0, max);
		}
		
		return p; 
	  
	   
	}
	
	static DoubleMatrix sigmoid(DoubleMatrix z)
	{
		// g = 1.0 ./ (1.0 + exp(-z));
		DoubleMatrix m = z.dup();
		for (int i = 0; i < m.getRows(); i++) 
		{
			for (int j = 0; j < m.getColumns(); j++) 
			{
				double y = m.get(i, j);
				double g = 1.0 / (1.0 + Math.exp(-y));
				m.put(i, j, g);
			}
		}
		return m;
	}

	/*static DoubleMatrix sigmoidAddOnes(DoubleMatrix z) 
	{
		// g = 1.0 ./ (1.0 + exp(-z));
		DoubleMatrix m = new DoubleMatrix(z.getRows(),
		    z.getColumns() + 1);
		for (int i = 0; i < m.getRows(); i++) {
			for (int j = 0; j < m.getColumns(); j++) {
				double g = 1.0;
				if (j != 0) {
					double y = z.get(i, j - 1);
					g = 1.0 / (1.0 + Math.exp(-y));
				}
				m.put(i, j, g);
			}
		}
		return m;
	}*/
	
	
	public static DoubleMatrix addOnes(DoubleMatrix z)
	{
		DoubleMatrix x = new DoubleMatrix(z.getRows(), z.getColumns()+1);
		
		for(int i=0; i<z.getRows(); i++)
		{
			for(int j=0; j<(z.getColumns()+1); j++)
			{
				double temp = 1.0; 
				if(j!=0)
				{
					temp = z.get(i, j-1);
				}
				x.put(i,j,temp);
			}
		}
		return x; 
	}
	
	public static DoubleMatrix addZeros(DoubleMatrix z)
	{
		DoubleMatrix x = new DoubleMatrix(z.getRows(), z.getColumns()+1);
		
		for(int i=0; i<z.getRows(); i++)
		{
			for(int j=0; j<(z.getColumns()+1); j++)
			{
				double temp = 0.0; 
				if(j!=0)
				{
					temp = z.get(i, j-1);
				}
				x.put(i,j,temp);
			}
		}
		return x; 
	}
	
	public static DoubleMatrix delColumn(DoubleMatrix z)
	{
		DoubleMatrix ret = new DoubleMatrix(z.getRows(), z.getColumns()-1);
		
		for(int i = 0; i < z.getRows(); i++)
		{
			for(int j = 1; j< z.getColumns(); j++)
			{
				double temp = z.get(i,j);
				ret.put(i,j-1,temp);
			}
		}
		
		
		return ret; 
	}
	
	public static DoubleMatrix addRow(DoubleMatrix z)
	{
		DoubleMatrix ret = new DoubleMatrix(z.getRows()+1,z.getColumns());
		
		for(int i =0; i<(z.getRows()+1) ; i++)
		{
			for(int j = 0; j<z.getColumns(); j++)
			{
				double temp = 1.0; 
				if(i!=0)
				{
					temp = z.get(i-1, j);
				}
				ret.put(i,j,temp);
				
			}
		}
		
		
		
		return ret;
	}
	
	
	/*public static DoubleMatrix reshape(DoubleMatrix A, int m, int n) 
	{
        int origM = A.getRows();
        int origN = A.getColumns();
        
        if(origM*origN != m*n)
        {
            throw new IllegalArgumentException("New matrix must be of same area as matix A");
        }
        
        DoubleMatrix B = new DoubleMatrix(m, n);
        double[] A1D = new double[origM * origN];

        int index = 0;
        for(int i = 0;i<origM;i++)
        {
            for(int j = 0;j<origN;j++)
            {
                A1D[index++] = A.get(i,j);
            }
        }

        index = 0;
        for(int i = 0;i<n;i++)
        {
            for(int j = 0;j<m;j++)
            {
            	double t = A1D[index++];
                B.put(j,i,t);
            }

        }
        return B;
    }*/
	
	public static void main(String[] args) 
	{
		
		int input_layer_size = 25344;
		int hidden_layer_size = 64;
		int num_labels = 4; 
		
		//load y.dat X.dat
		
		
		

	}

}
