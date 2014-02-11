#ifndef _XGBOOST_DATA_H_
#define _XGBOOST_DATA_H_

/*!
 * \file xgboost_data.h
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */

#include <vector>
#include "../utils/xgboost_utils.h"

namespace xgboost{
    namespace booster{
        /*! \brief interger type used in boost */
        typedef int bst_int;
        /*! \brief unsigned interger type used in boost */
        typedef unsigned bst_uint;
        /*! \brief float type used in boost */
        typedef float bst_float;
        /*! \brief debug option for booster */    
        const bool bst_debug = false;    
    };
};
namespace xgboost{
    namespace booster{
        /*! 
         * \brief feature matrix to store training instance, in sparse CSR format
         */
        class FMatrixS{
        public:
            /*! \brief one row of sparse feature matrix */
            struct Line{
                /*! \brief array of feature index */
                const bst_uint  *findex;
                /*! \brief array of feature value */
                const bst_float *fvalue;
                /*! \brief size of the data */
                bst_uint len;
            };
            /*! 
             * \brief remapped image of sparse matrix, 
             *  allows use a subset of sparse matrix, by specifying a rowmap
             */
            struct Image{
            public:
                Image( const FMatrixS &smat ):smat(smat), row_map( tmp_rowmap ){                
                }
                Image( const FMatrixS &smat, const std::vector<unsigned> &row_map )
                    :smat(smat), row_map(row_map){
                }
                /*! \brief get sparse part of current row */
                inline Line operator[]( size_t sidx ) const{
                    if( row_map.size() == 0 ) return smat[ sidx ];
                    else return smat[ row_map[ sidx ] ];
                }
            private:
                // used to set the simple case
                std::vector<unsigned> tmp_rowmap;
                const FMatrixS &smat;
                const std::vector<unsigned> &row_map;
            };
        public:
            // -----Note: unless needed for hacking, these fields should not be accessed directly -----
            /*! \brief row pointer of CSR sparse storage */
            std::vector<size_t>  row_ptr;
            /*! \brief index of CSR format */
            std::vector<bst_uint>   findex;
            /*! \brief value of CSR format */
            std::vector<bst_float>  fvalue;
        public:
            /*! \brief constructor */
            FMatrixS( void ){ this->Clear(); }
            /*! 
             * \brief get number of rows 
             * \return number of rows
             */
            inline size_t NumRow( void ) const{
                return row_ptr.size() - 1;
            }
            /*! 
             * \brief get number of nonzero entries
             * \return number of nonzero entries
             */
            inline size_t NumEntry( void ) const{
                return findex.size();
            }
            /*! \brief clear the storage */
            inline void Clear( void ){
                row_ptr.resize( 0 );
                findex.resize( 0 );
                fvalue.resize( 0 );
                row_ptr.push_back( 0 );
            }
            /*! 
             * \brief add a row to the matrix, but only accept features from fstart to fend
             *  \param feat sparse feature
             *  \param fstart start bound of feature
             *  \param fend   end bound range of feature
             *  \return the row id of added line
             */
            inline size_t AddRow( const Line &feat, unsigned fstart = 0, unsigned fend = UINT_MAX ){
                utils::Assert( feat.len >= 0, "sparse feature length can not be negative" );
                unsigned cnt = 0;
                for( unsigned i = 0; i < feat.len; i ++ ){
                    if( feat.findex[i] < fstart || feat.findex[i] >= fend ) continue;
                    findex.push_back( feat.findex[i] );
                    fvalue.push_back( feat.fvalue[i] );
                    cnt ++;
                }
                row_ptr.push_back( row_ptr.back() + cnt );
                return row_ptr.size() - 2;
            }

            /*! 
             * \brief add a row to the matrix, with data stored in STL container
             * \param findex feature index
             * \param fvalue feature value
             * \return the row id added line
             */
            inline size_t AddRow( const std::vector<bst_uint> &findex, 
                                  const std::vector<bst_float> &fvalue ){
                FMatrixS::Line l;
                utils::Assert( findex.size() == fvalue.size() );
                l.findex = &findex[0];
                l.fvalue = &fvalue[0];
                l.len = static_cast<bst_uint>( findex.size() );
                return this->AddRow( l );
            }
            /*! \brief get sparse part of current row */
            inline Line operator[]( size_t sidx ) const{
                Line sp;
                utils::Assert( !bst_debug || sidx < this->NumRow(), "row id exceed bound" );
                sp.len = static_cast<bst_uint>( row_ptr[ sidx + 1 ] - row_ptr[ sidx ] );
                sp.findex = &findex[ row_ptr[ sidx ] ];
                sp.fvalue = &fvalue[ row_ptr[ sidx ] ];
                return sp;
            }
        public:
            /*!
             * \brief save data to binary stream 
             *        note: since we have size_t in row_ptr, 
             *              the function is not consistent between 64bit and 32bit machine
             * \param fo output stream
             */
            inline void SaveBinary( utils::IStream &fo ) const{
                size_t nrow = this->NumRow();
                fo.Write( &nrow, sizeof(size_t) );
                fo.Write( &row_ptr[0], row_ptr.size() * sizeof(size_t) );
                if( findex.size() != 0 ){
                    fo.Write( &findex[0] , findex.size() * sizeof(bst_uint) );
                    fo.Write( &fvalue[0] , fvalue.size() * sizeof(bst_float) );
                }
            }
            /*!
             * \brief load data from binary stream 
             *        note: since we have size_t in row_ptr, 
             *              the function is not consistent between 64bit and 32bit machine
             * \param fi output stream
             */
            inline void LoadBinary( utils::IStream &fi ){
                size_t nrow;
                utils::Assert( fi.Read( &nrow, sizeof(size_t) ) != 0, "Load FMatrixS" );
                row_ptr.resize( nrow + 1 );
                utils::Assert( fi.Read( &row_ptr[0], row_ptr.size() * sizeof(size_t) ), "Load FMatrixS" );

                findex.resize( row_ptr.back() ); fvalue.resize( row_ptr.back() );                
                if( findex.size() != 0 ){
                    utils::Assert( fi.Read( &findex[0] , findex.size() * sizeof(bst_uint) ) , "Load FMatrixS" );
                    utils::Assert( fi.Read( &fvalue[0] , fvalue.size() * sizeof(bst_float) ), "Load FMatrixS" );
                }
            }
        };
    };    
};

#endif
