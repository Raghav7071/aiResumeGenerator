import React from 'react'
import Banner from '../components/Home/Banner'
import Hero from '../components/Home/Hero'
import Feather from "../components/Home/Feature"
import Testimonial from '../components/Home/Testimonial'
import CallToAction from '../components/Home/CallToAction'


const Home = () => {
  return (
    <div>
        <Banner />
        <Hero />
        <Feather />
        <Testimonial />
        <CallToAction />
       
         
    </div>
  )
}

export default Home