import warnings
warnings.filterwarnings("ignore")

import time
import os
from mongodb_retriever import MDBContextRetriever

def run_test_suite():
    """Run comprehensive test suite for hybrid search"""
    
    # Test cases organized by expected search method
    test_cases = [
        # Keyword search tests (exact matches)
        {
            "category": "KEYWORD SEARCH TESTS",
            "description": "Should trigger keyword search for exact matches",
            "tests": [
                ("Robin Hood", "Exact title match"),
                ("Buster Keaton", "Actor name"),
                ("train robbery", "Plot keywords"),
                ("Western", "Genre"),
                ("1920", "Year"),
                ("Charlie Chaplin", "Director/Actor")
            ]
        },
        # Semantic search tests (conceptual matches)
        {
            "category": "SEMANTIC SEARCH TESTS", 
            "description": "Should trigger semantic search for conceptual matches",
            "tests": [
                ("heroic outlaw tale", "Should find Robin Hood via semantic similarity"),
                ("comedic performer", "Should find comedy actors via semantic similarity"),
                ("romantic narrative", "Should find love stories via semantic similarity"),
                ("military conflict", "Should find war films via semantic similarity"),
                ("moving pictures", "Should find early animation via semantic similarity"),
                ("criminal underworld", "Should find crime films via semantic similarity")
            ]
        },
        # Edge cases and mixed scenarios
        {
            "category": "EDGE CASE TESTS",
            "description": "Complex queries that test hybrid behavior",
            "tests": [
                ("entertainer gets humiliated", "Should find 'He Who Gets Slapped' via semantic"),
                ("prehistoric creature cartoon", "Should find 'Gertie the Dinosaur' via semantic"),
                ("wordless humor", "Should find silent comedies via semantic"),
                ("religious dance drama", "Should find 'Salomè' via semantic"),
                ("creative genius story", "Should find invention-themed films via semantic"),
                ("xylophone zebra quantum", "Nonsense words - should force semantic search")
            ]
        }
    ]
    
    # Initialize retriever
    mongodb_uri = os.environ["ATLAS_URI"]
    retriever = MDBContextRetriever(mongodb_uri=mongodb_uri, k=3)
    
    print(f"\n🔧 TESTING SEARCH METHOD DETECTION:")
    print(f"If all queries return KEYWORD results, your text index is too broad.")
    print(f"Semantic search only triggers when keyword search returns 0 results.")
    print(f"The last test uses nonsense words to guarantee semantic search.\n")
    
    print(f"{'='*100}")
    print(f"🧪 HYBRID SEARCH TEST SUITE")
    print(f"Testing movies.json dataset with {len(sum([cat['tests'] for cat in test_cases], []))} queries")
    print(f"{'='*100}")
    
    total_tests = 0
    keyword_count = 0
    semantic_count = 0
    simple_count = 0
    
    for category in test_cases:
        print(f"\n{'🔥' if 'KEYWORD' in category['category'] else '🧠' if 'SEMANTIC' in category['category'] else '⚡'} {category['category']}")
        print(f"{category['description']}")
        print(f"{'='*80}")
        
        for query, expected in category["tests"]:
            total_tests += 1
            start_time = time.time()
            
            print(f"\n📋 TEST {total_tests}: '{query}'")
            print(f"Expected: {expected}")
            
            try:
                # Run the search
                docs = retriever.invoke(query)
                execution_time = (time.time() - start_time) * 1000
                
                # Analyze results
                if docs:
                    search_type = docs[0].metadata.get('search_type', 'UNKNOWN')
                    if search_type == 'KEYWORD':
                        keyword_count += 1
                    elif search_type == 'SEMANTIC':
                        semantic_count += 1
                    else:
                        simple_count += 1
                    
                    print(f"\n✅ RESULTS ({execution_time:.1f}ms):")
                    for i, doc in enumerate(docs[:3], 1):
                        title = doc.metadata.get('title', 'Unknown')
                        score = doc.metadata.get('score', 0)
                        search_method = doc.metadata.get('search_type', 'UNKNOWN')
                        icon = "📄" if search_method == "KEYWORD" else "🎯" if search_method == "SEMANTIC" else "🔍"
                        print(f"  {icon} #{i} [{search_method}] {title} (Score: {score:.4f})")
                else:
                    print(f"❌ No results found")
                    
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
            
            print(f"{'-'*60}")
            time.sleep(0.5)  # Brief pause between tests
    
    # Summary statistics
    print(f"\n{'='*100}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*100}")
    print(f"Total Tests: {total_tests}")
    print(f"📄 Keyword Search: {keyword_count} ({keyword_count/total_tests*100:.1f}%)")
    print(f"🎯 Semantic Search: {semantic_count} ({semantic_count/total_tests*100:.1f}%)")
    print(f"🔍 Simple Search: {simple_count} ({simple_count/total_tests*100:.1f}%)")
    print(f"{'='*100}")
    
    # Performance insights
    print(f"\n💡 INSIGHTS:")
    print(f"• Keyword search is fastest (~10-50ms) for exact matches")
    print(f"• Semantic search is slower (~100-500ms) but finds conceptual matches")
    print(f"• Simple search is the fallback when others fail")
    print(f"• Hybrid approach optimizes for both speed and accuracy")

def run_single_test(query):
    """Run a single test query"""
    mongodb_uri = os.environ["ATLAS_URI"]
    retriever = MDBContextRetriever(mongodb_uri=mongodb_uri, k=3)
    
    print(f"{'='*80}")
    print(f"🔍 SINGLE TEST: '{query}'")
    print(f"{'='*80}")
    
    start_time = time.time()
    docs = retriever.invoke(query)
    execution_time = (time.time() - start_time) * 1000
    
    if docs:
        search_type = docs[0].metadata.get('search_type', 'UNKNOWN')
        print(f"\n✅ RESULTS ({execution_time:.1f}ms using {search_type} search):")
        
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get('title', 'Unknown')
            score = doc.metadata.get('score', 0)
            search_method = doc.metadata.get('search_type', 'UNKNOWN')
            icon = "📄" if search_method == "KEYWORD" else "🎯" if search_method == "SEMANTIC" else "🔍"
            
            print(f"\n{icon} RESULT #{i} [{search_method}]")
            print(f"Title: {title}")
            print(f"Score: {score:.4f}")
            print(f"Plot: {doc.page_content[:200]}...")
    else:
        print(f"❌ No results found")

if __name__ == "__main__":
    # Choose test mode
    TEST_MODE = "full"  # Change to "single" to run just one test
    
    if TEST_MODE == "full":
        run_test_suite()
    else:
        # Test single query - change this to test different prompts
        test_query = "Robin Hood"  # Try: "adventure story", "clown performance", etc.
        run_single_test(test_query)