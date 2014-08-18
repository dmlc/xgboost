/*!
 * a tool to combine different set of features into binary buffer
 * not well organized code, but does it's job
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include "../src/io/simple_dmatrix-inl.hpp"
#include "../src/utils/utils.h"

using namespace xgboost;
using namespace xgboost::io;

// header in dataset
struct Header{
  FILE *fi;
  int   tmp_num;
  int   base;
  int   num_feat;
  // whether it's dense format
  bool  is_dense;
  bool  warned;
  
  Header( void ){ this->warned = false; this->is_dense = false; }
  
  inline void CheckBase( unsigned findex ){
    if( findex >= (unsigned)num_feat && ! warned ) {
      fprintf( stderr, "warning:some feature exceed bound, num_feat=%d\n", num_feat );
      warned = true;
    }
  }
};


inline int norm( std::vector<Header> &vec, int base = 0 ){
  int n = base;
  for( size_t i = 0; i < vec.size(); i ++ ){
    if( vec[i].is_dense ) vec[i].num_feat = 1;
    vec[i].base = n; n += vec[i].num_feat;
  }
  return n;        
}

inline void vclose( std::vector<Header> &vec ){
  for( size_t i = 0; i < vec.size(); i ++ ){
    fclose( vec[i].fi );
  }
}

inline int readnum( std::vector<Header> &vec ){
  int n = 0;
  for( size_t i = 0; i < vec.size(); i ++ ){
    if( !vec[i].is_dense ){
      utils::Assert( fscanf( vec[i].fi, "%d", &vec[i].tmp_num ) == 1, "load num" );
      n += vec[i].tmp_num;
    }else{
      n ++;
    }
  }
  return n;        
}

inline void vskip( std::vector<Header> &vec ){
  for( size_t i = 0; i < vec.size(); i ++ ){
    if( !vec[i].is_dense ){
      utils::Assert( fscanf( vec[i].fi, "%*d%*[^\n]\n" ) >= 0, "sparse" );
    }else{
      utils::Assert( fscanf( vec[i].fi, "%*f\n" ) >= 0, "dense" );
    }
  }
}

class DataLoader: public DMatrixSimple {
 public:
  // whether to do node and edge feature renormalization
  int rescale;
  int linelimit;
 public:
  FILE *fp, *fwlist, *fgroup, *fweight;
  std::vector<Header> fheader;
  DataLoader( void ){
    rescale = 0; 
    linelimit = -1;
    fp = NULL; fwlist = NULL; fgroup = NULL; fweight = NULL;
  }
 private:
  inline void Load( std::vector<SparseBatch::Entry> &feats, std::vector<Header> &vec ){
    SparseBatch::Entry e;
    for( size_t i = 0; i < vec.size(); i ++ ){
      if( !vec[i].is_dense ) { 
        for( int j = 0; j < vec[i].tmp_num; j ++ ){
          utils::Assert( fscanf ( vec[i].fi, "%u:%f", &e.findex, &e.fvalue ) == 2, "Error when load feat" );  
          vec[i].CheckBase( e.findex );
          e.findex += vec[i].base;
          feats.push_back(e);
        }
      }else{
        utils::Assert( fscanf ( vec[i].fi, "%f", &e.fvalue ) == 1, "load feat" );  
        e.findex = vec[i].base;
        feats.push_back(e);
      }
    }
  }
  inline void DoRescale( std::vector<SparseBatch::Entry> &vec ){
    double sum = 0.0;
    for( size_t i = 0; i < vec.size(); i ++ ){
      sum += vec[i].fvalue * vec[i].fvalue;
    } 
    sum = sqrt( sum );
    for( size_t i = 0; i < vec.size(); i ++ ){
      vec[i].fvalue /= sum;
    } 
  }    
 public:    
  // basically we are loading all the data inside
  inline void Load( void ){
    this->Clear();
    float label, weight = 0.0f;
    
    unsigned ngleft = 0, ngacc = 0;
    if( fgroup != NULL ){
      info.group_ptr.clear(); 
      info.group_ptr.push_back(0);
    }
    
    while( fscanf( fp, "%f", &label ) == 1 ){            
      if( ngleft == 0 && fgroup != NULL ){
        utils::Assert( fscanf( fgroup, "%u", &ngleft ) == 1, "group" );
      }
      if( fweight != NULL ){
        utils::Assert( fscanf( fweight, "%f", &weight ) == 1, "weight" );
      }
      
      ngleft -= 1; ngacc += 1;
      
      int pass = 1;
      if( fwlist != NULL ){
        utils::Assert( fscanf( fwlist, "%u", &pass ) ==1, "pass" );
      }
      if( pass == 0 ){
        vskip( fheader ); ngacc -= 1;
      }else{            
        const int nfeat = readnum( fheader );
        
        std::vector<SparseBatch::Entry> feats;
        
        // pairs 
        this->Load( feats, fheader );
        utils::Assert( feats.size() == (unsigned)nfeat, "nfeat" );
        if( rescale != 0 ) this->DoRescale( feats );
        // push back data :)
        this->info.labels.push_back( label );
        // push back weight if any
        if( fweight != NULL ){
          this->info.weights.push_back( weight );                    
        }
        this->AddRow( feats );
      }             
      if( ngleft == 0 && fgroup != NULL && ngacc != 0 ){
        info.group_ptr.push_back( info.group_ptr.back() + ngacc );
        utils::Assert( info.group_ptr.back() == info.num_row, "group size must match num rows" );
        ngacc = 0;
      }
      // linelimit
      if( linelimit >= 0 ) {
        if( -- linelimit <= 0 ) break;
      }
    }
    if( ngleft == 0 && fgroup != NULL && ngacc != 0 ){
      info.group_ptr.push_back( info.group_ptr.back() + ngacc );
      utils::Assert( info.group_ptr.back() == info.num_row, "group size must match num rows" );
    }
  }
  
};

const char *folder = "features";

int main( int argc, char *argv[] ){
  if( argc < 3 ){
    printf("Usage:xgcombine_buffer <inname> <outname> [options] -f [features] -fd [densefeatures]\n" \
           "options: -rescale -linelimit -fgroup <groupfilename> -wlist <whitelistinstance>\n");
    return 0; 
  }
  
  DataLoader loader;
  time_t start = time( NULL );
  
  int mode = 0;
  for( int i = 3; i < argc; i ++ ){        
    if( !strcmp( argv[i], "-f") ){
      mode = 0; continue;
    }
    if( !strcmp( argv[i], "-fd") ){
      mode = 2; continue;
    }
    if( !strcmp( argv[i], "-rescale") ){
      loader.rescale = 1; continue;
    }
    if( !strcmp( argv[i], "-wlist") ){
      loader.fwlist = utils::FopenCheck( argv[ ++i ], "r" ); continue;
    }
    if( !strcmp( argv[i], "-fgroup") ){
      loader.fgroup = utils::FopenCheck( argv[ ++i ], "r" ); continue;
    }
    if( !strcmp( argv[i], "-fweight") ){
      loader.fweight = utils::FopenCheck( argv[ ++i ], "r" ); continue;
    }
    if( !strcmp( argv[i], "-linelimit") ){
      loader.linelimit = atoi( argv[ ++i ] ); continue;
    }
    
    char name[ 256 ];
    sprintf( name, "%s/%s.%s", folder, argv[1], argv[i] );
    Header h;
    h.fi = utils::FopenCheck( name, "r" );
    
    if( mode == 2 ){
      h.is_dense = true; h.num_feat = 1;
      loader.fheader.push_back( h );
    }else{
      utils::Assert( fscanf( h.fi, "%d", &h.num_feat ) == 1, "num feat" );
      switch( mode ){
        case 0: loader.fheader.push_back( h ); break;
        default: ;
      }             
    }
  }
  loader.fp = utils::FopenCheck( argv[1], "r" );
  
  printf("num_features=%d\n", norm( loader.fheader ) ); 
  printf("start creating buffer...\n");
  loader.Load();
  loader.SaveBinary( argv[2] );
  // close files
  fclose( loader.fp );
  if( loader.fwlist != NULL ) fclose( loader.fwlist );    
  if( loader.fgroup != NULL ) fclose( loader.fgroup );    
  vclose( loader.fheader );
  printf("all generation end, %lu sec used\n", (unsigned long)(time(NULL) - start) );    
  return 0;
}
