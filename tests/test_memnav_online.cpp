#include "mem_nav.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <memory>
using namespace pipeann;
void require(bool x,const char*m){if(!x)throw std::runtime_error(m);}
template<class F> void until(F f,const char*m){auto end=std::chrono::steady_clock::now()+std::chrono::seconds(30);while(!f()){require(std::chrono::steady_clock::now()<end,m);std::this_thread::sleep_for(std::chrono::milliseconds(1));}}
int main(){
 auto distance=[](const uint8_t*a,const uint8_t*b){float d=0;for(int i=0;i<16;++i){require(b[i]==b[0],"torn private PQ row");float x=float(a[i])-b[i];d+=x*x;}return d;};
 MemNav nav(1024,640,16,16,distance,48);
 std::vector<uint64_t> tags;for(uint64_t i=0;tags.size()<2000;++i)if(MemNav::sample(i))tags.push_back(i);
 std::vector<uint64_t> current(512);std::vector<uint32_t> generations(512,1);
 for(uint32_t i=0;i<512;++i){current[i]=tags[i];nav.insert(i,current[i],std::vector<uint8_t>(16,uint8_t(i)));}nav.flush();
 auto pin=std::make_unique<NavReadDomain::Guard>(nav.read_guard());uint32_t oldslot=MemNav::invalid;
 for(uint32_t s=0;s<nav.cap;++s)if(nav.valid_identity(nav.identity(s))&&uint32_t(nav.identity(s))==0)oldslot=s;
 require(oldslot!=MemNav::invalid,"missing held slot");const auto *oldpq=nav.code(oldslot);auto epoch=nav.reader_generation();
 for(uint32_t i=0;i<100;++i)nav.erase_tag(current[i]);nav.flush();
 std::atomic<bool> done{false};std::thread maintenance([&]{nav.repair_and_reclaim();done=true;});
 until([&]{return nav.reader_generation()!=epoch;},"writer did not reach grace period");
 std::atomic<bool> run{true};std::atomic<uint64_t> reads{0};std::vector<std::thread> readers;
 for(int k=0;k<4;++k)readers.emplace_back([&,k]{uint8_t q[16];std::fill(q,q+16,k*31);while(run){
   auto batch=nav.candidates([&](const uint8_t*c){return distance(q,c);},32);
   for(auto p:batch){auto ident=nav.identity(p.second);if(!nav.valid_identity(ident))continue;
     auto expected=uint8_t(uint32_t(ident)+17*(MemNav::generation(ident)-1));
     for(uint32_t c=0;c<nav.chunks;++c)require(nav.code(p.second)[c]==expected,"PQ identity changed while result held");}
   ++reads;
 }});
 until([&]{return reads.load()>=100;},"new readers blocked behind old reader");
 std::this_thread::sleep_for(std::chrono::milliseconds(100));
 require(!done,"reclaimed while old reader pinned");for(int c=0;c<16;++c)require(oldpq[c]==0,"held old PQ overwritten");
 auto reads_while_old_pinned=reads.load();pin.reset();until([&]{return done.load();},"continuous new readers starved reclamation");maintenance.join();
 size_t next=512;
 for(uint32_t i=0;i<100;++i){current[i]=tags[next++];++generations[i];nav.insert(i,current[i],std::vector<uint8_t>(16,uint8_t(i+17*(generations[i]-1))));}nav.flush();
 for(uint32_t round=0;round<20;++round){
   for(uint32_t j=0;j<25;++j)nav.erase_tag(current[(round*25+j)%512]);
   nav.flush();nav.repair_and_reclaim();
   for(uint32_t j=0;j<25;++j){auto id=(round*25+j)%512;current[id]=tags[next++];++generations[id];nav.insert(id,current[id],std::vector<uint8_t>(16,uint8_t(id+17*(generations[id]-1))));}
   nav.flush();
 }
 run=false;for(auto&t:readers)t.join();std::ostringstream state;nav.audit(state);
 require(state.str().find("\"live\":512,")!=std::string::npos,"lost live slots");
 require(state.str().find("\"reclaimed\":600,")!=std::string::npos,"reuse not exercised");
 std::cout<<"MEMNAV_ONLINE_SAFETY_PASS {\"reads_while_old_pinned\":"<<reads_while_old_pinned<<",\"total_reads\":"<<reads<<",\"nav\":"<<state.str()<<"}"<<std::endl;
}
