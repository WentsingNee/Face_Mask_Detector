#define BOOST_TEST_MODULE Test

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(PassTest)
{
    BOOST_CHECK_EQUAL(4, 4);
}