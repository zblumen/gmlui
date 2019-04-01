import React from 'react';
import { render } from 'react-dom';
import Graph from './Graph.js';
import 'bootstrap/dist/css/bootstrap.css';
//import 'bootstrap/dist/css/bootstrap-theme.css'

class App extends React.Component {
	constructor(props) {
		super(props);
	
		this.state = {
		  cy_elements: {},
		};
	}

	componentDidMount() {
	/*	fetch('http://localhost:5000/cy_elements')
		.then(function(response) {
			return response.json();
		})
		.then(function(cy_elements) {
			console.log(JSON.stringify(cy_elements));
			this.setState({ cy_elements: cy_elements });
		});
	*/	
	fetch('http://localhost:5000/cy_elements')
	.then(response => {
        console.log(response);
        return response.json();
      })
      .then(data => {
        console.log(data);
       this.setState({ cy_elements : data });
      });
		//.then(response => response.json())
		//.then(cy_elements => this.setState({ cy_elements }));
	}
	render() {
		return (
			<div>
				<p>Node Calssification Using GraphSage Nueral Networks </p>
				<Graph elements = {this.state.cy_elements}/>
			</div>
		);
	}
}

render(<App />, document.getElementById('app'));