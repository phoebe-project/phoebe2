#include <iostream>
#include <vector>
#include <cmath>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

// Porting to Python 3
// Ref: http://python3porting.com/cextensions.html
#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
        static struct PyModuleDef moduledef = { \
          PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&moduledef);

  // adding missing declarations and functions
  #define PyString_Type PyBytes_Type
  #define PyString_AsString PyBytes_AsString
  #define PyString_Check PyBytes_Check
  #define PyInt_FromLong PyLong_FromLong
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
        ob = Py_InitModule3(name, methods, doc);
#endif


void convex_hull(double const* const points, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index);
                 
void inside_hull(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, npy_bool *inside);

void inside_hull_iterative(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, npy_bool *inside);
 
void inside_hull_sorted(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, npy_bool *inside);

void raise_exception(const std::string & str){
  std::cerr << str << std::endl;
  PyErr_SetString(PyExc_TypeError, str.c_str());
}


/* Perform a Graham scan and check which points are inside the hull

  Input: to-hull points, test points
  Output: hull, npy_bools if inside
*/
static PyObject *graham_scan_inside_hull(PyObject *dummy, PyObject *args)
{

  // 
  // Variable declaration
  //

  PyArrayObject *arr1, *arr2;
  
  // 
  // Argument parsing:
  // We need to parse the arguments, and then put the data
 
  if (!PyArg_ParseTuple(args, "O!O!", 
        &PyArray_Type, &arr1, 
        &PyArray_Type, &arr2)) {
    raise_exception("graham_scan_inside_hull::Problem reading arguments");
    return NULL;
  }
  
  double 
    *points = (double *)PyArray_DATA(arr1),     // pointer to data.
    *testpoints = (double *)PyArray_DATA(arr2); // pointer to data.
  
  int 
    n_points = PyArray_DIM(arr1, 0),
    t_points = PyArray_DIM(arr2, 0);
   
  // Create convex hull
  std::vector<double> myhull;
  
  int 
    h_points = 100,
    turn_index = 0;
  
  convex_hull(points, n_points, myhull, h_points, turn_index);    
 
  npy_intp  dims_inside[1] = {t_points};
  PyObject *inside_arr = PyArray_SimpleNew(1, dims_inside, NPY_BOOL);
  npy_bool *inside = (npy_bool*)PyArray_DATA((PyArrayObject *)inside_arr);
  
  // The following are different versions of the algorithm
  //inside_hull(testpoints, t_points, myhull, h_points, turn_index, inside);
  //inside_hull_iterative(testpoints, t_points, myhull, h_points, turn_index, inside);
  inside_hull_sorted(testpoints, t_points, myhull, h_points, turn_index, inside);
  
  // Store results 
  npy_intp dims_hull[2] = {h_points, 2};
  PyObject *hull_arr = PyArray_SimpleNew(2, dims_hull, NPY_DOUBLE);
  memcpy(PyArray_DATA((PyArrayObject *)hull_arr), &(myhull[0]), 2*h_points*sizeof(double));

  PyObject *res = PyTuple_New(2);
  PyTuple_SetItem(res, 0, hull_arr);
  PyTuple_SetItem(res, 1, inside_arr);

  return res;
}


int turn(double p1, double p2, double q1, double q2, double r1, double r2)
{
    /* Compute which way three points "bend".
     * 
     * 
     */
    double turn_value;
    int turn_flag = 0;
    
    turn_value = (q1-p1) * (r2-p2) - (r1-p1)*(q2-p2);
    
    if (turn_value < 0.0){
        turn_flag = -1;
    }
    else if (turn_value > 0.0){
        turn_flag =  1;
    }
    
    return turn_flag;   
}
    

void convex_hull(double const* const points, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index)
/* Perform a Graham scan of a set of points to find the convex hull.
 * 
 * In the first run, the lower part of the convex hull is found, in the second
 * part the upper part is found.
 * 
 * input: array of points [2*N]
 *        empty vector and initial guess of hull points
 * output: array of hull points [2, M] (M to be determined by this function)
 *         index of "turning point".
 */
{
    
    myhull.reserve(2*h_points);
    int msize = 0;
    int turn_flag;
    int xp = 0; // index to x-coordinate of points array
    int yp = n_points; // index to y-coordinate of points array
    int xhi1, yhi1, xhi2, yhi2;
    
    h_points = 2;
    
    // add first two points:
    myhull.push_back(points[0]);
    myhull.push_back(points[1]);
    myhull.push_back(points[2]);
    myhull.push_back(points[3]);
    
    // first run
    for(int i=2; i<n_points; i++){
        xp = 2*i;
        yp = 2*i+1;
        
        // if there are more than 2 (x,y) points in hull, check if the last one
        // turns in the correct direction
        if (myhull.size()/2 >= 2) {
            
            // if we turn the wrong way, we need to remove the point
            // from the hull. Then we need to check the turn for the
            // following two last points of the hull, untill exhaustion
            while (myhull.size()>=4){
            //for(int j=0; j<myhull.size()/2+3; j++){
                xhi1 = myhull.size()-4;
                yhi1 = myhull.size()-3;
                xhi2 = myhull.size()-2;
                yhi2 = myhull.size()-1;
                
                turn_flag = turn(myhull[xhi1], myhull[yhi1],
                                myhull[xhi2], myhull[yhi2],
                                points[xp], points[yp]);
                
                // Woopsy; if turn_flag does not equal one, it is not part of the
                // convex hull
                if (turn_flag < 1){
                    myhull.pop_back();
                    myhull.pop_back();
                    if (myhull.size()/2<2){
                        break;
                    }
                }
                else{
                    break;
                }
            }
        }
        
        // Add this point to the hull if it satisfies the convex hull criterion
        if ((myhull.size()/2==2) || (myhull[myhull.size()-2]!=points[xp]) || (myhull[myhull.size()-1]!=points[yp])){
            myhull.push_back(points[xp]);
            myhull.push_back(points[yp]);
        }
    }
    
    // Remember the turning point, and the current size of the hull (we need
    // to start over the search, but append the results to the previous vector)
    turn_index = myhull.size()/2;
    h_points = myhull.size()-2;
    
    // Second run
    for(int i=n_points-2; i>=0; i--){
        xp = 2*i;
        yp = 2*i+1;
        
        // if there are more than 2 (x,y) points in hull, check if the last one
        // turns in the correct direction
        msize = myhull.size()-h_points;
        if (msize/2 >= 2) {
            
            // if we turn the wrong way, we need to remove the point
            // from the hull. Then we need to check the turn for the
            // following two last points of the hull, untill exhaustion
            while ((int) myhull.size()>=h_points){
                xhi1 = myhull.size()-4;
                yhi1 = myhull.size()-3;
                xhi2 = myhull.size()-2;
                yhi2 = myhull.size()-1;
                
                turn_flag = turn(myhull[xhi1], myhull[yhi1],
                                myhull[xhi2], myhull[yhi2],
                                points[xp], points[yp]);
                // Woopsy; if turn_flag does not equal one, it is not part of the
                // convex hull
                if (turn_flag < 1){
                    myhull.pop_back();
                    myhull.pop_back();
                    
                    if ((myhull.size()-h_points)/2<2){
                        break;
                    }
                }
                else{
                    break;
                }
            }
        }
        
        msize = myhull.size() - h_points;
        // Add this point to the hull if it satisfies the convex hull criterion
        if ((msize/2==2) || (myhull[msize-2]!=points[xp]) || (myhull[msize-1]!=points[yp])){
            myhull.push_back(points[xp]);
            myhull.push_back(points[yp]);
        }
    }
    
    h_points = myhull.size()/2;
        
}


void inside_hull(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, npy_bool *inside)
/* Check which testpoints are inside a convex hull.
 * 
 */
{
    int xp = 0;
    int yp = 0;
    int myturn = 0;
    npy_bool skip = 0;
    int current_index = 0;
    int current_index2 = 0;
    double miny_hull =1e300;
    double maxy_hull = -1e300;
    
    // look for minimum and maximum y value, for chop-offs
    for (int i=0; i<h_points;i++){
        current_index = 2*i+1;
        if (myhull[current_index] < miny_hull){
            miny_hull = myhull[current_index];
        }
        if (myhull[current_index] > maxy_hull){
            maxy_hull = myhull[current_index];
        }
    }
    
    
    for (int i=0; i<n_points; i++){
        xp = 2*i;
        yp = 2*i+1;
        inside[i] = 1;
        
        // entirely on the left or right: the test point is outside the hull
        if ((testpoints[xp]<myhull[0]) || (myhull[2*turn_index-2]<testpoints[xp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = 0;
            continue;
        }
        if ((testpoints[yp]<miny_hull) || (maxy_hull<testpoints[yp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = 0;
            continue;
        }
        
        
        // below the lower hull? then outside the hull!
        //std::cout << "# test turn point: " << testpoints[xp]<<" "<< testpoints[yp] << std::endl;
        for (int j=2; j<h_points; j++){
            current_index = 2*(j-2);
            current_index2 = 2*(j-1);
            myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[current_index2], myhull[current_index2+1],
                    testpoints[xp], testpoints[yp]);
            //std::cout << "# .. " << myhull[2*(j-2)] << " "<< myhull[2*(j-2)+1]<< " "<<myhull[2*(j-1)]<< " "<<myhull[2*(j-1)+1] << " // "<<myturn << std::endl;
            if (myturn<1){
                inside[i] = 0;
                skip = 1;
                break;
            }
        }
        if (skip){
            skip = 0;
            continue;
        }
        
        // last points closes in on itself
        current_index = 2*(h_points-2);
        myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[0], myhull[1],
                    testpoints[xp], testpoints[yp]);
        if (myturn<1){
            inside[i] = 0;
        }
    }
    
}


int binary_search(std::vector<double>& A, int npoints, double key, int imin)
{
    int imax = npoints-1;
    int imid = 0;
    
    if (key < A[2*imin]){
        return imin;
    }
    else if (key > A[2*imax]){
        return imax;
    }
    
    while (imax >= imin){
        // calculate the midpoint for roughly equal partition
        imid = imin + ((imax-imin)/2);
        // key found at index imid
        if (A[2*imid] == key){
            return imid;
        }
        // determine which subarray to search
        else if (A[2*imid] < key){
            // change min index to search upper subarray
            imin = imid + 1;
        }
        // change max index to search lower subarray
        else{
            imax = imid - 1;
        }
    }
    // key was not found
    return imin;
}


void inside_hull_sorted(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, npy_bool *inside)
/* Check which testpoints are inside a convex hull.
 * 
 */
{
    int xp = 0;
    int yp = 0;
    int imid1 = 0;
    int myturn = 0;
    int current_index = 0;
    int current_index2 = 0;
    double miny_hull =1e300;
    double maxy_hull = -1e300;
    std::vector<double> imyhull; // uppper (inverted hull)
    
    // create a smaller version of the hull that only contains the upper
    // part, but in reversed order (which means from smaller x to larger x)
    
    imyhull.reserve(2*(h_points-turn_index+1));
    for (int i=h_points-1; i>=turn_index-1;i--){
        imyhull.push_back(myhull[2*i]);
        imyhull.push_back(myhull[2*i+1]);
    }
    
    
    // look for minimum and maximum y value, for easy chop-offs
    for (int i=0; i<h_points;i++){
        current_index = 2*i+1;
        if (myhull[current_index] < miny_hull){
            miny_hull = myhull[current_index];
        }
        if (myhull[current_index] > maxy_hull){
            maxy_hull = myhull[current_index];
        }
    }
    
    
    for (int i=0; i<n_points; i++){
        xp = 2*i;
        yp = 2*i+1;
        inside[i] = 1;
        
        // entirely on the left or right: the test point is outside the hull
        if ((testpoints[xp]<=myhull[0]) || (myhull[2*turn_index-2]<=testpoints[xp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = 0;
            continue;
        }
        if ((testpoints[yp]<=miny_hull) || (maxy_hull<=testpoints[yp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = 0;
            continue;
        }
        
        // below the lower hull? then outside the hull!
        // the n_points are sorted, so we can remember the best position
        // for the next run
        imid1 = binary_search(myhull, turn_index-1, testpoints[xp], 0);
        current_index = 2*(imid1-1);
        current_index2 = 2*(imid1);
        myturn = turn(myhull[current_index], myhull[current_index+1],
                      myhull[current_index2], myhull[current_index2+1],
                      testpoints[xp], testpoints[yp]);
        if (myturn<1){
            inside[i] = 0;
            continue;
        }
        
        // above the upper hull? then outside the hull!
        imid1 = binary_search(imyhull, imyhull.size()/2, testpoints[xp], 0);
        current_index = 2*(imid1-1);
        current_index2 = 2*(imid1);
        myturn = turn(imyhull[current_index], imyhull[current_index+1],
                      imyhull[current_index2], imyhull[current_index2+1],
                      testpoints[xp], testpoints[yp]);
        if (myturn>-1){
            inside[i] = 0;
        }
    }
    
}

void inside_hull_iterative(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, npy_bool *inside)
/* Check which testpoints are inside a convex hull.
 * 
 */
{
    int xp = 0;
    int yp = 0;
    int myturn = 0;
    npy_bool skip = 0;
    npy_bool *inside_treated;
    int resolution = 0;
    int current_index = 0;
    int current_index2 = 0;
    int split = 1;
    double miny_hull =1e300;
    double maxy_hull = -1e300;
    int total = 0;
    
    inside_treated = new npy_bool[n_points];
    
    // look for minimum and maximum y value, for chop-offs
    for (int i=0; i<h_points;i++){
        current_index = 2*i+1;
        if (myhull[current_index] < miny_hull){
            miny_hull = myhull[current_index];
        }
        if (myhull[current_index] > maxy_hull){
            maxy_hull = myhull[current_index];
        }
    }
    
    for (int i=0; i<n_points; i++){
        inside[i] = 1;
        inside_treated[i] = 0;
    }
    
    
    resolution = h_points/(4*split);
        
        
    for (int i=0; i<n_points; i++){
        if (inside_treated[i]){
            continue;
        }
        xp = 2*i;
        yp = 2*i+1;
        
        // below the lower hull? then outside the hull!
        for(int j=2;j<(4*split)+1;j++){
            current_index = 2*(j-2)*resolution;
            current_index2 = 2*(j-1)*resolution;
                            
            myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[current_index2], myhull[current_index2+1],
                    testpoints[xp], testpoints[yp]);
            if (myturn<1){
                skip = 1;
                break;
            }                        
        }

        if (skip){
            skip = 0;
            continue;
        }
        // last points closes in on itself
        current_index = 2*(4*split-1)*resolution;
        myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[0], myhull[1],
                    testpoints[xp], testpoints[yp]);
        
        if (myturn<1){
            continue;
        }
        // If we arrived here, the point is inside the hull!
        inside_treated[i] = 1;
        total = total+1;
        //std::cout << "split c" << split << std::endl;
    }
    
    for (int i=0; i<n_points; i++){
        xp = 2*i;
        yp = 2*i+1;
        
        if (inside_treated[i]){
            continue;
        }
        
        // entirely on the left or right: the test point is outside the hull
        if ((testpoints[xp]<myhull[0]) || (myhull[2*turn_index-2]<testpoints[xp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = 0;
            continue;
        }
        if ((testpoints[yp]<miny_hull) || (maxy_hull<testpoints[yp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = 0;
            continue;
        }
        
        
        // below the lower hull? then outside the hull!
        //std::cout << "# test turn point: " << testpoints[xp]<<" "<< testpoints[yp] << std::endl;
        for (int j=2; j<h_points; j++){
            current_index = 2*(j-2);
            current_index2 = 2*(j-1);
            myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[current_index2], myhull[current_index2+1],
                    testpoints[xp], testpoints[yp]);
            //std::cout << "# .. " << myhull[2*(j-2)] << " "<< myhull[2*(j-2)+1]<< " "<<myhull[2*(j-1)]<< " "<<myhull[2*(j-1)+1] << " // "<<myturn << std::endl;
            if (myturn<1){
                inside[i] = 0;
                skip = 1;
                break;
            }
        }
        if (skip){
            skip = 0;
            continue;
        }
        
        // last points closes in on itself
        current_index = 2*(h_points-2);
        myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[0], myhull[1],
                    testpoints[xp], testpoints[yp]);
        if (myturn<1){
            inside[i] = 0;
        }
    }
    delete inside_treated;
    
}
                 

/*
int main()
{   
     
     double array[2000];
     double testarray[2000];
     double x, y, u, v;
     npy_bool inside[1000];
     std::string line;
     std::ifstream myfile;
     myfile.open("test_random.txt");
     
     for(int i=0;i<1000;i++){
         
         myfile >> x >> y >> u >> v;
         array[i] = x;
         array[i+1000] = y;
         testarray[i] = u;
         testarray[i+1000] = v;
         
     }
     
     std::vector<double> myhull;
     int h_points = 10;
     int turn_index = 0;
     convex_hull(array, 1000, myhull, h_points, turn_index);
     inside_hull(testarray, 1000, myhull, h_points, turn_index, inside);
     
     
     // Attach the second run results to the first run results
     // bla bla
     for (int i=0; i<h_points; i++){
        std::cout << myhull[2*i] << " " << myhull[2*i+1] << std::endl;
     }
     std::cout << h_points << " " << turn_index << std::endl;
     
     std::cout << "===" << std::endl;
     
     for (int i=0; i<1000; i++){
       std::cout << inside[i] << std::endl;
     }
     return 0;
}
*/

static PyMethodDef Ceclipse_Methods[] = {
    {"graham_scan_inside_hull",
     graham_scan_inside_hull,
     METH_VARARGS,
     "Creating convex hull and do Graham scan"},
  
    {NULL,  NULL, 0, NULL} // terminator record
};

static char const *Ceclipse_Docstring = "Module for eclipsing with constructing a convex hull";

/* module initialization */
MOD_INIT(ceclipse) {

  PyObject *backend;

  MOD_DEF(backend, "ceclipse", Ceclipse_Docstring, Ceclipse_Methods)

  if (!backend) return MOD_ERROR_VAL;

  // Added to handle Numpy arrays
  // Ref:
  // * http://docs.scipy.org/doc/numpy-1.10.1/user/c-info.how-to-extend.html
  import_array();

  return MOD_SUCCESS_VAL(backend);
}

