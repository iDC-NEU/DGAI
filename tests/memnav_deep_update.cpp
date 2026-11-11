#include "dynamic_index.h"
#include "distance.h"
#include <fstream>
#include <chrono>
#include <sstream>
#include <unordered_set>
using namespace pipeann;
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char**argv){
 try {
  if(argc!=6)throw std::runtime_error("prefix full_base output rounds initial");
  std::string prefix=argv[1],base=argv[2],out=argv[3];uint32_t rounds=std::stoul(argv[4]),initial=std::stoul(argv[5]);
  uint32_t batch=initial/20;
  std::ifstream f(base,std::ios::binary);uint32_t n,d;f.read((char*)&n,4);f.read((char*)&d,4);
  check(n>=initial+rounds*batch&&d==96,"DEEP input shape");
  std::vector<float> data(size_t(n)*d);f.read((char*)data.data(),data.size()*4);check(bool(f),"DEEP input truncated");
  IndexBuildParameters p;p.set(64,128,384,1.2,24,true,4);
  DistanceL2Float dist;
  DynamicSSDIndex<float,uint32_t> dyn(p,prefix,prefix+"_out",&dist,Metric::L2,DECOUPLE_SEARCH,false);
  auto *idx=dyn._disk_index;
  {
    auto *pq=static_cast<PQNeighbor<float>*>(idx->nbr_handler);
    idx->memnav_=std::make_unique<MemNav>(initial+rounds*batch+1,128,32,pq->code_size(),
      [pq](const uint8_t*a,const uint8_t*b){return pq->code_distance(a,b);});
    idx->debug_memnav_seed_probe(data.data(),true);idx->memnav_.reset();
    std::cout<<"MEMNAV_EMPTY_FALLBACK_PASS"<<std::endl;
  }
  idx->enable_memnav(initial+rounds*batch+1,std::max<uint32_t>(1024,initial/100*3));
  auto audit=[&](uint32_t round,double insert_s,double merge_s){
    idx->memnav_->flush();idx->flush_all_pending_decouple_vectors();
    idx->debug_memnav_seed_probe(data.data(),false);
    idx->verify_graph_neighbor_ids();
    idx->reader_epoch_.quiescent([&]{
      uint64_t active=0;std::unordered_set<uint32_t> tags;
      for(uint32_t id=0;id<idx->cur_id;++id){auto loc=idx->id2loc(id);if(loc==kInvalidID)continue;
        check(idx->loc2id(loc)==id,"id/loc mismatch");if(idx->is_routing_only(id))continue;
        auto tag=idx->id2tag(id);check(tag>=round*batch&&tag<initial+round*batch,"wrong active tag range");
        check(tags.insert(tag).second,"duplicate tag");++active;
      }
      check(active==initial,"active count mismatch");
      uint64_t samples=0;
      for(uint32_t s=0;s<idx->memnav_->cap;++s){auto ident=idx->memnav_->identity(s);if(!idx->memnav_->valid_identity(ident))continue;
        auto id=uint32_t(ident);check(idx->id2loc(id)!=kInvalidID&&!idx->is_routing_only(id),"invalid nav seed");
        auto tag=idx->id2tag(id);check(tag>=round*batch&&tag<initial+round*batch,"deleted nav seed");++samples;
      }
      uint64_t expected=0;for(uint32_t tag=round*batch;tag<initial+round*batch;++tag)expected+=MemNav::sample(tag);
      check(samples==expected,"sample membership mismatch");
      std::ofstream o(out+"/round-"+std::to_string(round)+".json");
      o<<"{\"round\":"<<round<<",\"active\":"<<active<<",\"expected_samples\":"<<expected
       <<",\"insert_s\":"<<insert_s<<",\"merge_s\":"<<merge_s<<",\"nav\":";
      idx->memnav_->audit(o);o<<",\"maintenance_timing\":";idx->memnav_->timing.dump(o);o<<",\"status\":\"PASS\"}";o.flush();check(bool(o),"audit write failed");
    });
    std::cout<<"MEMNAV_ROUND_PASS "<<round<<std::endl;
  };
  audit(0,0,0);
  auto verify_legacy_seed_ids=[&](bool nonidentity_required){
    auto nav=std::move(idx->memnav_);
  idx->mem_index_=std::make_unique<Index<float,uint32_t>>(Metric::L2,d);
  std::vector<uint32_t> live;
  for(uint32_t id=0;id<idx->cur_id&&live.size()<4;++id)
    if(idx->id2loc(id)!=kInvalidID&&!idx->is_routing_only(id)&&(!nonidentity_required||idx->id2tag(id)!=id))live.push_back(id);
  check(live.size()==4,"four nonidentity legacy seeds required");
  auto &topo=*idx->mem_index_;topo._ep=0;topo._final_graph.resize(5);
  for(uint32_t i=0;i<4;++i)topo._location_to_tag[i]=idx->id2tag(live[i]);
  topo._location_to_tag[4]=nonidentity_required ? 0 : kInvalidID; // deleted/absent tag
  for(uint32_t i=0;i<5;++i)for(uint32_t j=0;j<5;++j)if(i!=j)topo._final_graph[i].push_back(j);
  idx->tag2id_cache_.reset(); // A separately constructed topology/load in this regression.
  auto legacy_seeds=idx->debug_memnav_seed_probe(data.data(),false);
  check(legacy_seeds.size()==live.size(),"legacy seed count/deleted-tag filter");
  for(auto id:legacy_seeds)check(std::find(live.begin(),live.end(),id)!=live.end(),"legacy API returned tag instead of ID");
  idx->mem_index_.reset();idx->memnav_=std::move(nav);
  };
  verify_legacy_seed_ids(false);
  std::cout<<"LEGACY_INITIAL_IDENTITY_SEEDS_PASS"<<std::endl;

  {
    idx->debug_memnav_seed_probe(data.data(),false);
  }
  // Exercise the real mem_search_ids_with_pq + seed filter while insert/merge run.
  // This is navigation-only: no disk ANN result/Recall/query-IO benchmark.
  std::atomic<bool> probing{true},probe_failed{false};std::atomic<uint64_t> probes{0};
  std::vector<std::thread> probe_threads;
  for(uint32_t k=0;k<4;++k)probe_threads.emplace_back([&,k]{
    try{while(probing){idx->debug_memnav_seed_probe(data.data()+size_t(700000+k)*d,false);++probes;std::this_thread::sleep_for(std::chrono::microseconds(200));}}
    catch(const std::exception&e){std::cerr<<"CONCURRENT_NAV_PROBE_FAILED "<<e.what()<<std::endl;probe_failed=true;probing=false;}
  });
  struct ProbeJoin {std::atomic<bool>&run;std::vector<std::thread>&threads;~ProbeJoin(){run=false;for(auto &t:threads)if(t.joinable())t.join();}} probe_join{probing,probe_threads};
  for(uint32_t r=1;r<=rounds;++r){
    idx->memnav_->flush();idx->memnav_->timing.reset();
    auto start=std::chrono::steady_clock::now();
    std::cout<<"MEMNAV_ROUND_BEGIN "<<r<<std::endl;
    for(uint32_t t=(r-1)*batch;t<r*batch;++t)dyn.lazy_delete(t);
    #pragma omp parallel for num_threads(24) schedule(dynamic,1)
    for(uint32_t t=initial+(r-1)*batch;t<initial+r*batch;++t)dyn.insert(data.data()+size_t(t)*d,t);
    auto end_insert=std::chrono::steady_clock::now();dyn.final_merge(24);
    auto end_merge=std::chrono::steady_clock::now();
    check(!probe_failed,"concurrent navigation failed");
    audit(r,std::chrono::duration<double>(end_insert-start).count(),std::chrono::duration<double>(end_merge-end_insert).count());
  }
  probing=false;for(auto&t:probe_threads)t.join();check(!probe_failed&&probes>0,"concurrent navigation probes failed");
  std::ofstream proof(out+"/concurrent-navigation.json");proof<<"{\"status\":\"PASS\",\"threads\":4,\"completed\":"<<probes<<",\"disk_query_evaluation\":false}";proof.close();
  idx->memnav_->flush();
  auto nav_seeds=idx->debug_memnav_seed_probe(data.data(),false);
  bool nonidentity=false;
  for(auto id:nav_seeds)nonidentity |= idx->id2tag(id)!=id;
  check(nonidentity,"regression requires reused internal IDs");
  verify_legacy_seed_ids(true);
  std::ofstream idproof(out+"/tag-id-regression.json");
  idproof<<"{\"status\":\"PASS\",\"initial_identity_tags\":true,\"nonidentity_ids\":true,\"legacy_live_seeds\":4,\"deleted_tag_filtered\":true,\"memnav_seed_count\":"<<nav_seeds.size()<<"}";idproof.close();
  std::cout<<"MEMNAV_TAG_ID_INTEGRATION_PASS"<<std::endl;
  // Save terminal native data; MemNav is deliberately rebuild-on-load v1.
  idx->flush_all_pending_decouple_vectors();
  std::ofstream done(out+"/COMPLETE.json");done<<"{\"status\":\"PASS\",\"rounds\":"<<rounds<<",\"query_evaluation\":false}";
  idx->memnav_.reset();
  std::cout<<"MEMNAV_DEEP_FULL_UPDATE_PASS"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<"MEMNAV_FAILED "<<e.what()<<std::endl;return 1;}
}
